import torch
import torch.nn as nn
from agents.components import TACO
import torch.nn.functional as F
import utils
from .components import RandomShiftsAug, ProprioceptiveEncoder, Critic
import itertools
import utils
from typing import Tuple, Optional, Dict, Any, Union

class TACOWrapper:
    """Wraps any classical algorithm with TACO enhancements"""

    def __init__(self, base_algorithm, encoder_lr, repr_dim, feature_dim,
                 hidden_dim, reward, multistep, latent_a_dim, curl, 
                 pretrained_path=None, freeze_encoder=False, no_enc_auxiliary_losses=False, optimizer_type="adam"):
        
        self.base_algorithm = base_algorithm
        self.obs_type = self.base_algorithm.obs_type
        self.obs_shape = self.base_algorithm.obs_shape
        self.action_shape = self.base_algorithm.action_shape
        self.device = self.base_algorithm.device
        self.update_every_steps = self.base_algorithm.update_every_steps
        self.use_tb = self.base_algorithm.use_tb
        self.num_expl_steps = self.base_algorithm.num_expl_steps
        self.stddev_schedule = self.base_algorithm.stddev_schedule
        self.stddev_clip = self.base_algorithm.stddev_clip
        self.freeze_encoder = self.base_algorithm.freeze_encoder

        self.reward = reward
        self.curl = curl
        self.multistep = multistep
        self.optimizer_type = optimizer_type.lower()

        # 🔥 MONKEY PATCH: Sostituisci update_critic del base_algorithm
        self._monkey_patch_base_algorithm()


        ### A heuristics to choose the dimensionality of latent actions
        if latent_a_dim == 'none':
            latent_a_dim = int(self.action_shape[0]*1.25)+1
        print(f"Using latent action dimension: {latent_a_dim}")
        self.latent_a_dim = latent_a_dim
        ### Create action embeddings
        self.act_tok = utils.ActionEncoding(self.action_shape[0], self.latent_a_dim, multistep)
        self.base_algorithm.act_tok = self.act_tok

        # Initialize encoder
        if hasattr(self.base_algorithm, 'encoder') and not isinstance(self.base_algorithm.encoder, nn.Identity): # Assume that every vision based algorithm has an encoder attribute, and prioprio based algorithm always use Identity
            self.encoder = self.base_algorithm.encoder
            self.load_encoder = False
        else:
            assert self.obs_type == 'proprio_obs', "With image observations, an encoder must be provided in the base algorithm."
            self.encoder = ProprioceptiveEncoder(self.obs_shape, repr_dim).to(self.device)
            self.base_algorithm.encoder = self.encoder  
            self.load_encoder = self.base_algorithm.load_encoder
            self.base_algorithm.feature_dim = feature_dim
            self.base_algorithm.build_actor()  # We need to rebuild actor to match the new encoder output size
        self.base_algorithm.build_critic(target=self.base_algorithm.has_critic_target)  # We need to rebuild critic to match the new encoder output size


        self.TACO = TACO(self.encoder.repr_dim, feature_dim, self.action_shape, latent_a_dim, hidden_dim, self.act_tok, self.encoder, self.multistep, self.device).to(self.device)
        self.freeze_encoder = freeze_encoder
        self.no_enc_auxiliary_losses = no_enc_auxiliary_losses
        ### State & Action Encoders
        parameters = itertools.chain(self.encoder.parameters(),
                                     self.act_tok.parameters(),
        )
        
        # Selezione dell'optimizer
        if self.optimizer_type == "adam":
            optimizer_class = torch.optim.Adam
        elif self.optimizer_type == "sgd":
            optimizer_class = torch.optim.SGD
        else:
            raise ValueError(f"Optimizer type '{self.optimizer_type}' not supported. Use 'adam' or 'sgd'.")
        
        if not self.freeze_encoder:
            self.encoder_opt = optimizer_class(parameters, lr=encoder_lr)
            if not self.no_enc_auxiliary_losses:
                self.taco_opt = optimizer_class(self.TACO.parameters(), lr=encoder_lr)

        self.base_algorithm.build_optimizers()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # data augmentation
        if hasattr(self.base_algorithm, 'aug'):
            self.aug = self.base_algorithm.aug
        else:
            if self.obs_type == 'pixel_obs':
                self.aug = RandomShiftsAug(pad=4)
            else:
                self.aug = nn.Identity()

        if pretrained_path is not None and pretrained_path != 'none' and self.load_encoder:
            self._load_components(pretrained_path)
        if freeze_encoder:
            self._freeze_encoder()
        
         
        
        self.pretrained_path = pretrained_path
        self.train()
        utils.ColorPrint.green("Using TACO Wrapper")

    def get_meta_specs(self):
        return self.base_algorithm.get_meta_specs()

    def init_meta(self):
        return self.base_algorithm.init_meta()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return self.base_algorithm.update_meta(meta, global_step, time_step, finetune)
    
    def _monkey_patch_base_algorithm(self):
        """Substitute original update_critic method of base_algorithm with TACO version"""
        # Save the original method for future reference (if needed)
        if hasattr(self.base_algorithm, 'update_critic'):            

            # Replace with the wrapper method
            self.base_algorithm.update_critic = self.update_critic
            
            utils.ColorPrint.blue("🔧 Monkey patched base_algorithm.update_critic with TACO version")
        
        if hasattr(self.base_algorithm, 'build_critic'):

            self.base_algorithm.build_critic = self.build_critic

            utils.ColorPrint.blue("🔧 Monkey patched base_algorithm.build_critic to match TACO action encoder output size")
    
    def build_critic(self, target: bool):
        """Build the critic network."""
        print("Building TACO-wrapped critic...")
         # Critic (note: DrQV2 uses raw action_dim, not latent_a_dim)
        self.base_algorithm.critic = Critic(
            self.encoder.repr_dim,
            self.latent_a_dim,  # new custom latent dim
            self.base_algorithm.feature_dim,
            self.base_algorithm.hidden_dim
        ).to(self.device)
        
        if target:
            self.base_algorithm.critic_target = Critic(
                self.encoder.repr_dim,
                self.latent_a_dim,  # Raw action dimension
                self.base_algorithm.feature_dim,
                self.base_algorithm.hidden_dim
            ).to(self.device)

            self.base_algorithm.critic_target.load_state_dict(self.base_algorithm.critic.state_dict())

    def _load_components(self, models_path: Union[str, Dict[str, Any]]):
        utils.ColorPrint.blue(f"Loading pretrained model from: {models_path}")

        if isinstance(models_path, str):
            checkpoint = torch.load(models_path, map_location=self.device, weights_only=False)
        else:
            checkpoint = models_path

        print("Loading encoder weights...")
        if 'agent' in checkpoint:
            agent_state = checkpoint['agent']
            self.TACO.load_state_dict(agent_state.TACO.state_dict())
            utils.ColorPrint.green("✓  Loaded pretrained TACO model.")
            if isinstance(self.encoder, ProprioceptiveEncoder):
                self.encoder.load_state_dict(agent_state.encoder.state_dict())
                utils.ColorPrint.green("✓  Loaded pretrained encoder.")
            self.act_tok.load_state_dict(agent_state.act_tok.state_dict())
            utils.ColorPrint.green("✓  Loaded pretrained action tokenizer.")
        else:
            self.TACO.load_state_dict(checkpoint['taco'])
            utils.ColorPrint.green("✓  Loaded pretrained TACO model.")
            
            if isinstance(self.encoder, ProprioceptiveEncoder):
                self.encoder.load_state_dict(checkpoint['encoder'])
                utils.ColorPrint.green("✓  Loaded pretrained encoder.")
            
            self.act_tok.load_state_dict(checkpoint['act_tok'])
            utils.ColorPrint.green("✓  Loaded pretrained action tokenizer.")
                
    def _freeze_encoder(self):
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.act_tok.eval()
        for param in self.act_tok.parameters():
            param.requires_grad = False
        
        utils.ColorPrint.blue("🔒 Encoder components frozen")

    def train(self, training=True):
        self.training = training
        self.base_algorithm.train(training)
        self.encoder.train(training)
        self.TACO.train(training)
        self.act_tok.train(training)

    def act(self, obs, meta, step, eval_mode):
        return self.base_algorithm.act(obs, meta, step, eval_mode)
    
    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.base_algorithm.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.base_algorithm.critic_target(next_obs, next_action, self.act_tok)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.base_algorithm.critic(obs, action, self.act_tok)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        if not self.freeze_encoder:
            self.encoder_opt.zero_grad(set_to_none=True) # TODO check encoder were really frozen, I should put a if freeze don't optimize to be 100% sure
        self.base_algorithm.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.base_algorithm.critic_opt.step()
        if not self.freeze_encoder:
            self.encoder_opt.step()

        return metrics
    

    def update_taco(self, obs, action, action_seq, next_obs, reward):
        metrics = dict()
        
        obs_anchor = self.aug(obs.float())
        obs_pos = self.aug(obs.float())
        z_a = self.TACO.encode(obs_anchor)
        z_pos = self.TACO.encode(obs_pos, ema=True)
        ### Compute CURL loss
        if self.curl:
            logits = self.TACO.compute_logits(z_a, z_pos)
            labels = torch.arange(logits.shape[0]).long().to(self.device)
            curl_loss = self.cross_entropy_loss(logits, labels)
        else:
            curl_loss = torch.tensor(0.)
        
        ### Compute action encodings
        action_en = self.TACO.act_tok(action, seq=False) 
        action_seq_en = self.TACO.act_tok(action_seq, seq=True)
        
        ### Compute reward prediction loss
        if self.reward:
            reward_pred = self.TACO.reward(torch.concat([z_a, action_seq_en], dim=-1))
            reward_loss = F.mse_loss(reward_pred, reward)
        else:
            reward_loss = torch.tensor(0.)
        
        ### Compute TACO loss
        next_z = self.TACO.encode(self.aug(next_obs.float()), ema=True)
        curr_za = self.TACO.project_sa(z_a, action_seq_en) 
        logits = self.TACO.compute_logits(curr_za, next_z)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        taco_loss = self.cross_entropy_loss(logits, labels)
        
        if not self.freeze_encoder:
            self.taco_opt.zero_grad()
            (taco_loss + curl_loss + reward_loss).backward()
            self.taco_opt.step()
        if self.use_tb:
            metrics['reward_loss']  = reward_loss.item()
            metrics['curl_loss'] = curl_loss.item()
            metrics['taco_loss']  = taco_loss.item()
            metrics['total_loss'] = taco_loss.item() + curl_loss.item() + reward_loss.item()
        return metrics
    

    def update(self, replay_iter, step):
        metrics = dict()
        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, action_seq, reward, discount, next_obs, r_next_obs = utils.to_torch(
            batch, self.device)

        metrics.update(self.base_algorithm.update(iter([batch]), step)) # create a new iterator for the batch maybe is not the best way but works
        # TODO change the signature of base_algorithm.update to accept directly obs, action, etc.

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()
        
        if self.no_enc_auxiliary_losses:
            return metrics
        
        metrics.update(self.update_taco(obs, action, action_seq, r_next_obs, reward))
        return metrics
    
    
    def evaluate_taco(self, obs, action, action_seq, next_obs, reward):
        with torch.no_grad():
            metrics = dict()
            metrics['batch_reward'] = reward.mean().item()
            
            obs_anchor = self.aug(obs.float())
            obs_pos = self.aug(obs.float())
            z_a = self.TACO.encode(obs_anchor)
            z_pos = self.TACO.encode(obs_pos, ema=True)
            ### Compute CURL loss
            if self.curl:
                logits = self.TACO.compute_logits(z_a, z_pos)
                labels = torch.arange(logits.shape[0]).long().to(self.device)
                curl_loss = self.cross_entropy_loss(logits, labels)
            else:
                curl_loss = torch.tensor(0.)
            
            ### Compute action encodings
            action_en = self.TACO.act_tok(action, seq=False) 
            action_seq_en = self.TACO.act_tok(action_seq, seq=True)
            
            ### Compute reward prediction loss
            if self.reward:
                reward_pred = self.TACO.reward(torch.concat([z_a, action_seq_en], dim=-1))
                reward_loss = F.mse_loss(reward_pred, reward)
            else:
                reward_loss = torch.tensor(0.)
            
            ### Compute TACO loss
            next_z = self.TACO.encode(self.aug(next_obs.float()), ema=True)
            curr_za = self.TACO.project_sa(z_a, action_seq_en) 
            logits = self.TACO.compute_logits(curr_za, next_z)
            labels = torch.arange(logits.shape[0]).long().to(self.device)
            taco_loss = self.cross_entropy_loss(logits, labels)
            
            if self.use_tb:
                metrics['reward_loss']  = reward_loss.item()
                metrics['curl_loss'] = curl_loss.item()
                metrics['taco_loss']  = taco_loss.item()
                metrics['total_loss'] = taco_loss.item() + curl_loss.item() + reward_loss.item()
        return metrics
