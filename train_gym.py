import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'osmesa'  # or whatever default you want

from pathlib import Path

import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
from dm_env import specs

from envs import gym_env as gym_env
import wandb
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True


def make_wrapped_agent(obs_spec, action_spec, agent_cfg, wrapper_cfg=None):
    # Handle agent configuration from global config
    agent_cfg.obs_shape = obs_spec.shape
    agent_cfg.action_shape = action_spec.shape

    if wrapper_cfg is not None:
        print(f"Using wrapper: {wrapper_cfg._target_}")
        # Instantiate base agent first
        base_agent = hydra.utils.instantiate(agent_cfg)
        
        # Pass required parameters to wrapper
        return hydra.utils.instantiate(
            wrapper_cfg,
            base_algorithm=base_agent,
            obs_shape=obs_spec.shape,
            action_shape=action_spec.shape)
    
    return hydra.utils.instantiate(agent_cfg)

# def make_wrapped_agent(obs_spec, action_spec, agent_cfg, wrapper_cfg=None):
#     """Create wrapped agent with dynamic algorithm selection"""
    
#     # Handle nested structure: if agent_cfg has 'agent' key, use that
#     if hasattr(agent_cfg, 'agent') and agent_cfg.agent is not None:
#         actual_agent_cfg = agent_cfg.agent
#     else:
#         actual_agent_cfg = agent_cfg
    
#     # Debug: Print the configuration
#     print(f"Agent config _target_: {actual_agent_cfg.get('_target_', 'NOT FOUND')}")
    
#     # Set obs_shape and action_shape directly on cfg
#     actual_agent_cfg.obs_shape = obs_spec.shape
#     actual_agent_cfg.action_shape = action_spec.shape
    
#     # Instantiate base algorithm
#     agent = hydra.utils.instantiate(actual_agent_cfg)
    
#     # Check if wrapper is specified
#     if wrapper_cfg is not None:
#         print(f"Using wrapper: {wrapper_cfg._target_}")
        
#         # Get device from agent_cfg or use default
#         device = actual_agent_cfg.device if hasattr(actual_agent_cfg, 'device') else 'cuda'
        
#         # Wrap with TACO (or other wrapper)
#         agent = hydra.utils.instantiate(
#             wrapper_cfg,
#             base_algorithm=agent,
#             obs_shape=obs_spec.shape,
#             action_shape=action_spec.shape,
#             device=device
#         )
    
#     return agent


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        # Get observation and action specs for the agent
        if self.obs_type=='proprio_obs':
            obs_spec = gym_env.proprio_obs_spec(self.train_env)
        elif self.obs_type=='pixel_obs':
            obs_spec = gym_env.pixel_obs_spec(self.train_env)
        else:
            raise ValueError(f"Unknown obs_type {self.cfg.obs_type}")  
        action_spec = gym_env.action_spec(self.train_env)
        
        # Check if wrapper is specified in config
        wrapper_cfg = None
        if hasattr(cfg, 'wrapper') and cfg.wrapper is not None and cfg.wrapper != 'none':
            wrapper_cfg = cfg.wrapper
        
        # Passa l'intera configurazione, non cfg.agent
        self.agent = make_wrapped_agent(obs_spec, action_spec, self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        # Initialize encoder saving flags and step thresholds
        self.encoder_step_thresholds = getattr(cfg, 'encoder_checkpoint_steps')
        print(f"Encoder checkpoints will be saved at steps: {self.encoder_step_thresholds}")
        self.encoder_saved_flags = {step: False for step in self.encoder_step_thresholds}

        if cfg.use_wandb:
            if cfg.wandb_id is not None and cfg.wandb_id != "none":
                wandb.init(
                    id=cfg.wandb_id,
                    resume='must',
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name,
                    tags=cfg.wandb_tag.split('_') if cfg.wandb_tag and cfg.wandb_tag != "none" else None,
                    sync_tensorboard=True,
                    mode='online')
            else:
                wandb.init(
                    config=OmegaConf.to_container(cfg, resolve=True),
                    project=cfg.wandb_project,
                    name=cfg.wandb_run_name,
                    tags=cfg.wandb_tag.split('_') if cfg.wandb_tag and cfg.wandb_tag != "none" else None,
                    sync_tensorboard=True,
                    mode='online')
                
            wandb.run.save()
            
            

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        self.train_env = gym_env.make(self.cfg.task_name, self.cfg.frame_stack,
                                self.cfg.action_repeat, self.cfg.seed, self.cfg.resolution, self.cfg.random_init, self.cfg.random_goal)
        self.eval_env = gym_env.make(self.cfg.task_name, self.cfg.frame_stack,
                                self.cfg.action_repeat, self.cfg.seed, self.cfg.resolution, self.cfg.random_init, self.cfg.random_goal)

        sample_time_step = self.train_env.reset()
        proprio_shape = sample_time_step.proprio_obs.shape

        # create replay buffer
        data_specs = (gym_env.pixel_obs_spec(self.train_env),
                      gym_env.action_spec(self.train_env),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'),
                      specs.Array(proprio_shape, np.float32, 'proprio_obs'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')
        
        self.obs_type = utils.OBS_KEY_REGISTRY.get(self.cfg.obs_type, self.cfg.obs_type)
        print(f"Observation key for the replay buffer:")
        utils.ColorPrint.blue(f"{self.obs_type}")

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.multistep, self.cfg.discount, obs_type=self.obs_type)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)
        
    def check_and_save_step_checkpoints(self, current_reward=None):
        """Check if we need to save encoder checkpoints based on step thresholds"""
        for step_threshold in self.encoder_step_thresholds:
            # Check if we've reached this step threshold and haven't saved yet
            if (self.global_step >= step_threshold and 
                not self.encoder_saved_flags[step_threshold]):
                
                self.encoder_saved_flags[step_threshold] = True
                self.save_encoder_checkpoint(step_threshold, current_reward)

    def save_encoder_checkpoint(self, step_threshold, current_reward=None):
        """Save encoder checkpoint with informative naming and replay buffer"""
        import shutil
        
        # Create encoder state dict similar to pretrain_MT_multiheads.py
        encoder_state = {
            'encoder': self.agent.encoder.state_dict(),
            'taco': self.agent.TACO.state_dict(),
            'act_tok': self.agent.act_tok.state_dict(),
            'args': {
                'env_name': self.cfg.task_name,
                'seed': self.cfg.seed,
                'global_step': self.global_step,
                'global_episode': self.global_episode,
                'step_threshold': step_threshold,
                'current_reward': current_reward
            },
            'global_step': self.global_step,
            'global_episode': self.global_episode,
            'step_threshold': step_threshold,
            'current_reward': current_reward
        }
        
        # Create informative folder name
        step_str = f"step{step_threshold}"
        episode_str = f"ep{self.global_episode}"
        reward_str = f"rew{current_reward:.0f}" if current_reward is not None else "rewN/A"
        
        checkpoint_folder_name = f"checkpoint_{self.cfg.task_name}_{step_str}_{episode_str}_{reward_str}"
        checkpoint_folder = self.work_dir / checkpoint_folder_name
        
        # Create checkpoint folder
        checkpoint_folder.mkdir(exist_ok=True)
        
        # Save encoder model
        model_filename = f"encoder_{self.cfg.task_name}_{step_str}_{episode_str}_{reward_str}.pt"
        model_path = checkpoint_folder / model_filename
        torch.save(encoder_state, model_path)
        
        # Copy replay buffer folder
        replay_buffer_source = self.work_dir / 'buffer'
        replay_buffer_dest = checkpoint_folder / 'buffer'
        
        if replay_buffer_source.exists():
            # Copy the entire buffer directory
            shutil.copytree(replay_buffer_source, replay_buffer_dest, dirs_exist_ok=True)
            print(f'Replay buffer copied to: {replay_buffer_dest}')
        
        self.save_snapshot_with_name(f"{step_str}_{episode_str}_{reward_str}")

        print(f'Checkpoint saved: {checkpoint_folder} (Step: {step_threshold}, Reward: {current_reward if current_reward is not None else 0})')

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward, success = 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    if self.obs_type=='proprio_obs':
                        obs = time_step.proprio_obs
                    elif self.obs_type=='pixel_obs':
                        obs = time_step.pixel_obs
   
                    action = self.agent.act(obs,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        avg_episode_reward = total_reward / episode

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)
            log('global_frame', self.global_frame)
        
        if self.cfg.use_wandb:
            wandb.log({
                'eval/episode_reward': avg_episode_reward,
                'eval/episode_length': step * self.cfg.action_repeat / episode,
                'eval/episode': self.global_episode,
                'eval/step': self.global_step,
                'eval/success_rate': success / episode,
                'buffer_size': len(self.replay_storage),
                'global_frame': self.global_frame
            })


    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.pixel_obs)
        metrics = None

        while train_until_step(self.global_step):
            self.check_and_save_step_checkpoints(episode_reward)

            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)
                    # Log also on wandb:
                    if self.cfg.use_wandb:
                        wandb.log({
                            'fps': episode_frame / elapsed_time,
                            'total_time': total_time,
                            'episode_reward': episode_reward,
                            'episode_length': episode_frame,
                            'episode': self.global_episode,
                            'buffer_size': len(self.replay_storage),
                            'step': self.global_step
                        })


                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                self.train_video_recorder.init(time_step.pixel_obs)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                if self.obs_type=='proprio_obs':
                        obs = time_step.proprio_obs
                elif self.obs_type=='pixel_obs':
                    obs = time_step.pixel_obs
                action = self.agent.act(obs,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

                if self.cfg.use_wandb:
                    metrics.update({'global_step': self.global_step,
                            'episode': self.global_episode,
                            'buffer_size': len(self.replay_storage),
                            'step': self.global_step,
                            'global_frame': self.global_frame
                            })
                    wandb.log(metrics)


            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            self.train_video_recorder.record(time_step.pixel_obs)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def save_snapshot_with_name(self, name):
        snapshot = self.work_dir / f'snapshot_{name}.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            print("loading snapshot: ", f)
            payload = torch.load(f, weights_only=False)
        for k, v in payload.items():
            if self.cfg.checkpoint==True:
                if k != '_global_step' and k != '_global_episode':
                    self.__dict__[k] = v
            else:   
                self.__dict__[k] = v
        print(f'loaded snapshot: {snapshot} in checkpoint_mode {self.cfg.checkpoint}')


@hydra.main(config_path='cfgs', config_name='config_gym_new', version_base='1.1')
def main(cfg):
    from pathlib import Path
    if cfg.use_wandb:
        wandb.tensorboard.patch(root_logdir=str(Path.cwd()))
    from train_gym import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if cfg.checkpoint:
        assert snapshot.exists(), f'checkpoint {snapshot} does not exist!'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
