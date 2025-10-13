"""
DrQV2 Agent

DrQV2 (Data-Regularized Q-Learning V2) agent for visual RL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import utils

from .base_agent import BaseAgent
from .networks import RandomShiftsAug, ImageEncoder, Actor, Critic


class DrQV2Agent(BaseAgent):
    """
    DrQV2 agent for image-based reinforcement learning.
    
    Uses data augmentation and regularization for sample-efficient
    learning from visual observations.
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        device: torch.device,
        lr: float,
        feature_dim: int,
        hidden_dim: int,
        critic_target_tau: float,
        num_expl_steps: int,
        update_every_steps: int,
        stddev_schedule: str,
        stddev_clip: float,
        use_tb: bool,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = False
    ):
        """
        Initialize DrQV2 agent.
        
        Args:
            obs_shape: Shape of image observations (C, H, W)
            action_shape: Shape of action space
            device: Device for computation
            lr: Learning rate
            feature_dim: Dimension of feature representations
            hidden_dim: Dimension of hidden layers
            critic_target_tau: Soft update coefficient
            num_expl_steps: Number of exploration steps
            update_every_steps: Frequency of updates
            stddev_schedule: Schedule for action noise
            stddev_clip: Clipping value for action noise
            use_tb: Whether to use tensorboard
            pretrained_path: Path to pretrained checkpoint
            freeze_encoder: Whether to freeze encoder
        """
        super().__init__(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            lr=lr,
            use_tb=use_tb,
            pretrained_path=pretrained_path,
            freeze_encoder=freeze_encoder
        )
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.critic_target_tau = critic_target_tau
        self.num_expl_steps = num_expl_steps
        self.update_every_steps = update_every_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        
        # Build networks
        self.build_networks()
        
        # Build optimizers
        self.build_optimizers()
        
        # Loss functions
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        # Data augmentation
        self.aug = RandomShiftsAug(pad=4)
        
        # Load pretrained weights if provided
        if pretrained_path is not None and pretrained_path.lower() != 'none':
            print(f"Loading pretrained model from {pretrained_path}, "
                  f"freeze encoder: {freeze_encoder}")
            self.load_pretrained(pretrained_path, None, freeze_encoder)
        else:
            print("No pretrained model provided, initializing from scratch.")
        
        # Set training mode
        self.train()
        self.critic_target.train()
    
    def build_networks(self):
        """Build all neural networks for the agent."""
        # Image encoder
        self.encoder = ImageEncoder(self.obs_shape, self.feature_dim).to(self.device)
        
        # Actor
        self.actor = Actor(
            self.encoder.repr_dim,
            self.action_shape,
            self.feature_dim,
            self.hidden_dim
        ).to(self.device)
        
        # Critic (note: DrQV2 uses raw action_dim, not latent_a_dim)
        self.critic = Critic(
            self.encoder.repr_dim,
            self.action_shape[0],  # Raw action dimension
            self.feature_dim,
            self.hidden_dim
        ).to(self.device)
        
        self.critic_target = Critic(
            self.encoder.repr_dim,
            self.action_shape[0],  # Raw action dimension
            self.feature_dim,
            self.hidden_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
    
    def build_optimizers(self):
        """Build optimizers for all components."""
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
    
    def _load_checkpoint_components(
        self,
        checkpoint: Dict[str, Any],
        freeze_encoder: bool
    ):
        """
        Load encoder from checkpoint.
        
        Args:
            checkpoint: Loaded checkpoint dictionary
            freeze_encoder: Whether encoder will be frozen
        """
        if 'encoder' in checkpoint:
            self.encoder.load_state_dict(checkpoint['encoder'])
            utils.ColorPrint.green("✓ Loaded encoder from checkpoint")
        else:
            utils.ColorPrint.yellow("! No encoder found in checkpoint")
    
    def _freeze_components(self):
        """Freeze encoder."""
        self._frozen_fingerprints = {
            'encoder': self._get_model_fingerprint(self.encoder),
        }
        
        self.encoder.eval()
        
        # Disable gradients
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        utils.ColorPrint.blue("🔒 Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder."""
        if self._frozen_fingerprints is not None:
            for param in self.encoder.parameters():
                param.requires_grad = True
            
            self.encoder.train()
            
            super().unfreeze_encoder()
    
    def train(self, training: bool = True):
        """Set training mode."""
        super().train(training)
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
    
    def act(self, obs: torch.Tensor, step: int, eval_mode: bool):
        """
        Select action given observation.
        
        Args:
            obs: Observation tensor
            step: Current step
            eval_mode: Whether in evaluation mode
            
        Returns:
            Action as numpy array
        """
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
    
    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        step: int
    ) -> Dict[str, float]:
        """
        Update critic networks.
        
        Args:
            obs: Current observations
            action: Actions taken
            reward: Rewards received
            discount: Discount factors
            next_obs: Next observations
            step: Current step
            
        Returns:
            Dictionary of metrics
        """
        metrics = dict()
        
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            
            # DrQV2: No action tokenizer
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)
        
        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        
        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
        
        # Optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()
        
        return metrics
    
    def update_actor(self, obs: torch.Tensor, step: int) -> Dict[str, float]:
        """
        Update actor network.
        
        Args:
            obs: Observations
            step: Current step
            
        Returns:
            Dictionary of metrics
        """
        metrics = dict()
        
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)
        
        actor_loss = -Q.mean()
        
        # Optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        
        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
        
        return metrics
    
    def update(self, replay_iter, step: int) -> Dict[str, float]:
        """
        Full update for DrQV2 agent.
        
        Args:
            replay_iter: Iterator over replay buffer
            step: Current step
            
        Returns:
            Dictionary of metrics
        """
        metrics = dict()
        
        if step % self.update_every_steps != 0:
            return metrics
        
        # Get batch
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device
        )
        
        # Augment and encode
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)
        
        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()
        
        # Update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step)
        )
        
        # Update actor
        metrics.update(self.update_actor(obs.detach(), step))
        
        # Soft update target critic
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )
        
        return metrics
