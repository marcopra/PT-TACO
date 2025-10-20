"""
DrQV2 Agent

DrQV2 (Data-Regularized Q-Learning V2) agent for visual RL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, Union
import utils

from .base_agent import BaseAgent
from .components import RandomShiftsAug, ImageEncoder, Actor, Critic


class DrQV2Agent(BaseAgent):
    """
    DrQV2 agent for image-based reinforcement learning.
    
    Uses data augmentation and regularization for sample-efficient
    learning from visual observations.
    """
    
    def __init__(
        self,
        obs_type: str,
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
        load_encoder: bool = True,
        load_actor: bool = False,
        load_critic: bool = False,
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

        self.obs_type = utils.OBS_KEY_REGISTRY.get(obs_type, obs_type)
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.critic_target_tau = critic_target_tau
        self.num_expl_steps = num_expl_steps
        self.update_every_steps = update_every_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.has_critic_target = True

        self.load_encoder = load_encoder
        self.load_actor = load_actor
        self.load_critic = load_critic

        # Build networks
        self.build_networks()
        
        # Build optimizers
        self.build_optimizers()
        
        # Loss functions
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        # Data augmentation
        if self.obs_type == 'pixels':
            self.aug = RandomShiftsAug(pad=4)
        else:
            self.aug = nn.Identity()

        if self.pretrained_path is not None and self.pretrained_path != "none":
            self._load_components(
                self.pretrained_path,
                load_encoder=load_encoder,
                load_actor=load_actor,
                load_critic=load_critic
            )
        if freeze_encoder:
            self._freeze_encoder()
            
        # Set training mode
        self.train()
        self.critic_target.train()
    
    def build_networks(self):
        """Build all neural networks for the agent."""
        self.build_encoder()
        self.build_actor()
        self.build_critic(target=self.has_critic_target)
    
    
    
    def unfreeze_encoder(self):
        raise NotImplementedError("un freeze to be implemented in subclass")
        """Unfreeze encoder."""
        if self._frozen_fingerprints is not None:
            for param in self.encoder.parameters():
                param.requires_grad = True
            
            self.encoder.train()
            
            super().unfreeze_encoder()
    
    def train(self, training: bool = True):
        """Set training mode."""
        super().train(training)
        if self.encoder_opt is not None:
            self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
    
    def act(self, obs: torch.Tensor, meta: Tuple, step: int, eval_mode: bool):
        """
        Select action given observation.
        
        Args:
            obs: Observation tensor
            meta: Meta information (used only for compatibility)
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
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
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
        obs, action, action_seq, reward, discount, next_obs, r_next_obs = utils.to_torch(
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
