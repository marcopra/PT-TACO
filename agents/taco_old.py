import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import itertools


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)
        
class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())
        
        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class TACO(nn.Module):
    """
    TACO Constrastive loss
    """

    def __init__(self, repr_dim, feature_dim, action_shape, latent_a_dim, hidden_dim, act_tok, encoder, multistep, device):
        super(TACO, self).__init__()

        self.multistep = multistep
        self.encoder = encoder
        self.device = device
        
        a_dim = action_shape[0]

        self.proj_sa = nn.Sequential(
            nn.Linear(feature_dim + latent_a_dim*multistep, hidden_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )
        
        self.act_tok = act_tok
        
        self.proj_s = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())
        
        self.reward = nn.Sequential(
            nn.Linear(feature_dim+latent_a_dim*multistep, hidden_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        self.W = nn.Parameter(torch.rand(feature_dim, feature_dim))
        self.apply(utils.weight_init)
    
    def encode(self, x, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.proj_s(self.encoder(x))
        else:
            z_out = self.proj_s(self.encoder(x))
        return z_out
    
    def project_sa(self, s, a):
        x = torch.concat([s,a], dim=-1)
        return self.proj_sa(x)
        
    def compute_logits(self, z_a, z_pos):
        """
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits
    
    def load_checkpoint(self, state_dict):
        """
        Load TACO weights from checkpoint with selective loading.
        Loads all TACO networks and reward network (if available and compatible).
        
        Args:
            state_dict: The state dictionary from the checkpoint
        """
        loaded_components = []
        failed_components = []
        excluded_components = []
        
        # First, try to load the complete state dict with strict=False
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        if unexpected_keys:
            utils.ColorPrint.yellow(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
        
        # Check what was loaded successfully
        all_keys = set(state_dict.keys())
        missing_keys_set = set(missing_keys)
        loaded_keys = all_keys - missing_keys_set
        
        # Categorize loaded components
        component_prefixes = ['encoder.', 'act_tok.', 'proj_s.', 'proj_sa.', 'W']
        
        for prefix in component_prefixes:
            if prefix == 'W':
                if 'W' in loaded_keys:
                    loaded_components.append('W')
            else:
                component_keys = [k for k in loaded_keys if k.startswith(prefix)]
                if component_keys:
                    component_name = prefix.rstrip('.')
                    loaded_components.append(component_name)
        
        # Special handling for reward network
        reward_keys = [k for k in state_dict.keys() if k.startswith('reward.')]
        if reward_keys:
            try:
                reward_state_dict = {k.replace('reward.', ''): v for k, v in state_dict.items() if k.startswith('reward.')}
                
                # Check if the reward network architecture is compatible
                try:
                    self.reward.load_state_dict(reward_state_dict, strict=True)
                    loaded_components.append('reward')
                    utils.ColorPrint.green(f"✓ Loaded reward network with {len(reward_keys)} parameters")
                except (RuntimeError, ValueError) as e:
                    utils.ColorPrint.yellow(f"! Reward network architecture mismatch: {e}")
                    utils.ColorPrint.yellow("  Reinitializing reward network from scratch")
                    self.reward.apply(utils.weight_init)
                    failed_components.append('reward (architecture mismatch)')
                    
            except Exception as e:
                utils.ColorPrint.red(f"✗ Failed to load reward network: {e}")
                utils.ColorPrint.yellow("  Reinitializing reward network from scratch")
                self.reward.apply(utils.weight_init)
                failed_components.append('reward')
        else:
            utils.ColorPrint.yellow("! No reward network found in checkpoint, keeping initialized weights")
        
        # Report on missing components
        if missing_keys:
            missing_non_reward = [k for k in missing_keys if not k.startswith('reward.')]
            if missing_non_reward:
                utils.ColorPrint.red(f"✗ Missing critical TACO components: {missing_non_reward}")
                failed_components.extend([k.split('.')[0] for k in missing_non_reward])
        
        # Summary
        utils.ColorPrint.blue(f"TACO Checkpoint Loading Summary:")
        if loaded_components:
            utils.ColorPrint.blue(f"  ✓ Loaded: {', '.join(loaded_components)}")
        if failed_components:
            utils.ColorPrint.blue(f"  ✗ Failed/Reinitialized: {', '.join(failed_components)}")
            
        # Log successful loading for required components
        required_components = ['encoder', 'act_tok', 'proj_s', 'proj_sa', 'W']
        loaded_required = [c for c in required_components if c in loaded_components]
        if len(loaded_required) == len(required_components):
            utils.ColorPrint.green("✓ All required TACO components loaded successfully")
        else:
            missing_required = [c for c in required_components if c not in loaded_components]
            utils.ColorPrint.red(f"✗ Missing required components: {', '.join(missing_required)}")

    
class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, latent_a_dim, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + latent_a_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + latent_a_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action, act_tok=None):
        if act_tok is not None:
            action = act_tok(action)
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2
    

class TACOAgent:
    def __init__(self, obs_shape, action_shape, device, lr, encoder_lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 reward, multistep, latent_a_dim, curl, pretrained_path=None, freeze_encoder=False, no_taco=False, optimizer_type="adam"):
    
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        
        self.reward = reward
        self.multistep = multistep
        self.curl = curl
        self.optimizer_type = optimizer_type.lower()

        ### A heuristics to choose the dimensionality of latent actions
        if latent_a_dim == 'none':
            latent_a_dim = int(action_shape[0]*1.25)+1
        ### Create action embeddings
        self.act_tok = utils.ActionEncoding(action_shape[0], latent_a_dim, multistep)
        self.encoder = Encoder(obs_shape, feature_dim).to(device)
        
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)
        self.critic = Critic(self.encoder.repr_dim, latent_a_dim, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, latent_a_dim,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.TACO = TACO(self.encoder.repr_dim, feature_dim, action_shape, latent_a_dim, hidden_dim, self.act_tok, self.encoder, multistep, device).to(device)
        self.freeze_encoder = freeze_encoder
        self.no_taco = no_taco
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
            if not self.no_taco:
                self.taco_opt = optimizer_class(self.TACO.parameters(), lr=encoder_lr)
        self.actor_opt = optimizer_class(self.actor.parameters(), lr=lr)
        self.critic_opt = optimizer_class(self.critic.parameters(), lr=lr)
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        if pretrained_path is None or pretrained_path.lower() == 'none':
            print("No pretrained model provided, initializing from scratch.")
        else:
            print(f"Loading pretrained model from {pretrained_path}, freeze encoder: {freeze_encoder}")
            self.load_pretrained(pretrained_path, None, self.freeze_encoder)
            
        self.pretrained_path = pretrained_path
        self.train()
        self.critic_target.train()

    def load_pretrained(self, model_path, map_location=None, freeze_encoder=False):
        """
        Load a pretrained TACO model from a saved checkpoint.
        
        Args:
            model_path: Path to the saved model checkpoint
            map_location: Optional device mapping for torch.load
            freeze_encoder: Whether to freeze the loaded components
        
        Returns:
            dict: The original training arguments
        """
        if map_location is None:
            map_location = self.device
            
        utils.ColorPrint.blue(f"Loading pretrained model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=map_location, weights_only=False)
        
        # Use TACO's load_checkpoint method for all TACO components
        if 'taco' in checkpoint:
            self.TACO.load_checkpoint(checkpoint['taco'])
        else:
            utils.ColorPrint.yellow("! No TACO state found in checkpoint")
        
        # Handle freezing if requested
        if freeze_encoder:
            self._frozen_fingerprints = {
                'encoder': self._get_model_fingerprint(self.encoder),
                'taco': self._get_model_fingerprint(self.TACO),
                'act_tok': self._get_model_fingerprint(self.act_tok)
            }
            
            # Set to eval mode and disable gradients
            self.encoder.eval()
            self.TACO.eval()
            self.act_tok.eval()
            
            # Disable gradients for all parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.TACO.parameters():
                param.requires_grad = False
            for param in self.act_tok.parameters():
                param.requires_grad = False
                
            utils.ColorPrint.blue("🔒 Encoder components frozen")
        
        utils.ColorPrint.green("✓ Pretrained model loading completed")
        return checkpoint.get('args', {})  # Return the saved args for reference
    
    def _get_model_fingerprint(self, model):
        """Generate a unique fingerprint for model parameters"""
        return {name: param.data.clone() for name, param in model.named_parameters()}
        
    def _check_frozen_models(self):
        """Check if frozen models have been modified"""
        if not hasattr(self, '_frozen_fingerprints'):
            raise ValueError("No frozen fingerprints found. Did you load a pretrained model with freeze_encoder=True?")
            
        for model_name, fingerprint in self._frozen_fingerprints.items():
            model = getattr(self, model_name.upper() if model_name == 'taco' else model_name)
            current_fingerprint = self._get_model_fingerprint(model)
            
            for param_name, stored_param in fingerprint.items():
                current_param = current_fingerprint[param_name]
                assert torch.all(torch.eq(current_param, stored_param)), f"Parameter {param_name} in {model_name} has changed when it should be frozen!"

    def unfreeze_encoder(self):
        """Riattiva i gradienti per i modelli congelati"""
        if hasattr(self, '_frozen_fingerprints'):
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.TACO.parameters():
                param.requires_grad = True
            for param in self.act_tok.parameters():
                param.requires_grad = True
            
            self.encoder.train()
            self.TACO.train()
            self.act_tok.train()
            
            del self._frozen_fingerprints
            self.freeze_encoder = False
            
    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if not self.freeze_encoder:
            self.encoder.train(training)
            self.TACO.train()

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action, self.act_tok)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action, self.act_tok)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        if not self.freeze_encoder:
            self.encoder_opt.zero_grad(set_to_none=True) # TODO check encoder were really frozen, I should put a if freeze don't optimize to be 100% sure
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if not self.freeze_encoder:
            self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action, self.act_tok)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

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

        # augment
        obs_en = self.aug(obs.float())
        next_obs_en = self.aug(next_obs.float())
        # encode
        obs_en = self.encoder(obs_en)
        with torch.no_grad():
            next_obs_en = self.encoder(next_obs_en)
        
        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs_en, action, reward, discount, next_obs_en, step))

        # update actor
        metrics.update(self.update_actor(obs_en.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)
        
        if self.no_taco:
            metrics['reward_loss']  = torch.tensor(0.)
            metrics['curl_loss'] = torch.tensor(0.)
            metrics['taco_loss']  = torch.tensor(0.)
            return metrics
            
        metrics.update(self.update_taco(obs, action, action_seq, r_next_obs, reward))       
        
        # # save model periodically
        # if step % 10000 == 0:
        #     save_path = f"/home/mprattico/Pretrain-TACO/test_model/model_step_{step}.pt"
        #     self.save(save_path, step)
        #     print(f"Model saved at step {step} to {save_path}")
        # # Verify that frozen models haven't been modified
        # if self.freeze_encoder and self.pretrained_path is not None and self.pretrained_path.lower() != 'none':
        #     self._check_frozen_models()
        #     #check if the model corresponds to the pretrained one reloading from the pretrained path
        #     pretrained_checkpoint = torch.load(self.pretrained_path, map_location=self.device)
        #     pretrained_encoder_state = pretrained_checkpoint['encoder']
        #     pretrained_taco_state = pretrained_checkpoint['taco'] 
        #     pretrained_act_tok_state = pretrained_checkpoint['act_tok']
            
        #     # Compare current model states with pretrained states
        #     current_encoder_state = self.encoder.state_dict()
        #     current_taco_state = self.TACO.state_dict()
        #     current_act_tok_state = self.act_tok.state_dict()
            
        #     # Check encoder parameters
        #     for key in pretrained_encoder_state:
        #         if not torch.all(torch.eq(pretrained_encoder_state[key], current_encoder_state[key])):
        #             raise ValueError(f"Encoder parameter {key} has changed when it should be frozen!")
            
        #     # Check TACO parameters
        #     for key in pretrained_taco_state:
        #         if not torch.all(torch.eq(pretrained_taco_state[key], current_taco_state[key])):
        #             raise ValueError(f"TACO parameter {key} has changed when it should be frozen!")
            
        #     # Check act_tok parameters
        #     for key in pretrained_act_tok_state:
        #         if not torch.all(torch.eq(pretrained_act_tok_state[key], current_act_tok_state[key])):
        #             raise ValueError(f"act_tok parameter {key} has changed when it should be frozen!")
            
        #     # print("All frozen models are unchanged from the pretrained model.")
            

        return metrics
    
    def save(self, save_path, step):
        """
        Save the model state to a file.
        
        Args:
            save_path: Path to save the model state
            step: Current training step (for naming purposes)
        """
        state = {
            'encoder': self.encoder.state_dict(),
            'taco': self.TACO.state_dict(),
            'act_tok': self.act_tok.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'args': {
                'step': step,
                'freeze_encoder': self.freeze_encoder,
                'no_taco': self.no_taco
            }
        }
        torch.save(state, save_path)
        print(f"Model saved to {save_path}")
        
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
                print("curl_loss", curl_loss)
            else:
                curl_loss = torch.tensor(0.)
            
            ### Compute action encodings
            action_en = self.TACO.act_tok(action, seq=False) 
            action_seq_en = self.TACO.act_tok(action_seq, seq=True)
            
            ### Compute reward prediction loss
            if self.reward:
                reward_pred = self.TACO.reward(torch.concat([z_a, action_seq_en], dim=-1))
                reward_loss = F.mse_loss(reward_pred, reward)

                # Average percentage of reward prediction error
                
                metrics['avg_rew_pred_error_percentage'] = torch.mean(torch.abs(reward_pred - reward) / (reward + 1e-6)).item() 
                error = reward_pred - reward
                metrics['log_cosh'] = torch.mean(torch.log(torch.cosh(error + 1e-12))).item()
                threshold = 1e-3  # puoi settarlo in base al tuo dominio
                mask = reward.abs() > threshold
                metrics['rel_error_filtered'] = torch.mean(
                    torch.abs(reward_pred[mask] - reward[mask]) / (reward[mask] + 1e-6)
                ).item()
                numerator = torch.abs(reward_pred - reward)
                denominator = torch.abs(reward_pred) + torch.abs(reward) + 1e-6
                metrics['smape'] = torch.mean(2.0 * numerator / denominator).item()

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
    
    def change_device(self, new_device):
        """
        Move all model components to a new device.
        
        Args:
            new_device: The target device (e.g., 'cuda:0', 'cpu')
        """
        # Update device attribute
        self.device = new_device
        
        # Move all model components to new device
        self.encoder = self.encoder.to(new_device)
        self.actor = self.actor.to(new_device)
        self.critic = self.critic.to(new_device)
        self.critic_target = self.critic_target.to(new_device)
        self.TACO = self.TACO.to(new_device)
        self.act_tok = self.act_tok.to(new_device)
        
        # Update TACO's internal device reference
        self.TACO.device = new_device
        
        # Move optimizers' state to new device if they exist
        if hasattr(self, 'encoder_opt') and self.encoder_opt.state:
            for state in self.encoder_opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(new_device)
        
        if hasattr(self, 'taco_opt') and self.taco_opt.state:
            for state in self.taco_opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(new_device)
        
        if hasattr(self, 'actor_opt') and self.actor_opt.state:
            for state in self.actor_opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(new_device)
        
        if hasattr(self, 'critic_opt') and self.critic_opt.state:
            for state in self.critic_opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(new_device)
        
        # Update frozen fingerprints if they exist
        if hasattr(self, '_frozen_fingerprints'):
            for model_name, fingerprint in self._frozen_fingerprints.items():
                for param_name, param_tensor in fingerprint.items():
                    fingerprint[param_name] = param_tensor.to(new_device)
        
        print(f"All components moved to device: {new_device}")
