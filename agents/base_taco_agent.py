"""
Base TACO Agent

This class provides common functionality for all agents that use TACO
(Temporal Action Contrastive Learning) for representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple
import itertools
import utils

from .base_agent import BaseAgent
from .networks import RandomShiftsAug, Actor, Critic, TACOModule


class BaseTACOAgent(BaseAgent):
    """
    Base class for agents using TACO contrastive learning.
    
    Provides:
    - TACO module initialization and management
    - Action tokenization
    - TACO loss computation
    - Reward prediction
    - Common update patterns for TACO-based agents
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        device: torch.device,
        lr: float,
        encoder_lr: float,
        feature_dim: int,
        hidden_dim: int,
        critic_target_tau: float,
        num_expl_steps: int,
        update_every_steps: int,
        stddev_schedule: str,
        stddev_clip: float,
        use_tb: bool,
        reward: bool,
        multistep: int,
        latent_a_dim: Any,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = False,
        no_taco: bool = False,
        optimizer_type: str = "adam"
    ):
        """
        Initialize base TACO agent.
        
        Args:
            obs_shape: Shape of observation space
            action_shape: Shape of action space
            device: Device for computation
            lr: Learning rate for actor and critic
            encoder_lr: Learning rate for encoder and TACO
            feature_dim: Dimension of feature representations
            hidden_dim: Dimension of hidden layers
            critic_target_tau: Soft update coefficient for target networks
            num_expl_steps: Number of exploration steps
            update_every_steps: Frequency of updates
            stddev_schedule: Schedule for action noise
            stddev_clip: Clipping value for action noise
            use_tb: Whether to use tensorboard
            reward: Whether to use reward prediction
            multistep: Number of action steps for TACO
            latent_a_dim: Dimension of latent action embeddings
            pretrained_path: Path to pretrained checkpoint
            freeze_encoder: Whether to freeze encoder
            no_taco: Whether to disable TACO loss
            optimizer_type: Type of optimizer ('adam' or 'sgd')
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
        
        self.encoder_lr = encoder_lr
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.critic_target_tau = critic_target_tau
        self.num_expl_steps = num_expl_steps
        self.update_every_steps = update_every_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.reward = reward
        self.multistep = multistep
        self.no_taco = no_taco
        self.optimizer_type = optimizer_type.lower()
        
        # Heuristic for latent action dimension
        if latent_a_dim == 'none':
            latent_a_dim = int(action_shape[0] * 1.25) + 1
        self.latent_a_dim = latent_a_dim
        
        # Create action tokenizer
        self.act_tok = utils.ActionEncoding(
            action_shape[0],
            latent_a_dim,
            multistep
        )
        
        # Data augmentation
        self.aug = RandomShiftsAug(pad=4)
        
        # Loss functions
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # Networks and optimizers will be built by subclasses
        self.encoder: Optional[nn.Module] = None
        self.actor: Optional[Actor] = None
        self.critic: Optional[Critic] = None
        self.critic_target: Optional[Critic] = None
        self.TACO: Optional[TACOModule] = None
        
        # Optimizers
        self.encoder_opt: Optional[torch.optim.Optimizer] = None
        self.actor_opt: Optional[torch.optim.Optimizer] = None
        self.critic_opt: Optional[torch.optim.Optimizer] = None
        self.taco_opt: Optional[torch.optim.Optimizer] = None
        
    def _get_optimizer_class(self) -> type:
        """Get optimizer class based on optimizer_type."""
        if self.optimizer_type == "adam":
            return torch.optim.Adam
        elif self.optimizer_type == "sgd":
            return torch.optim.SGD
        else:
            raise ValueError(
                f"Optimizer type '{self.optimizer_type}' not supported. "
                f"Use 'adam' or 'sgd'."
            )
    
    def build_optimizers(self):
        """Build optimizers for TACO agent components."""
        optimizer_class = self._get_optimizer_class()
        
        # Encoder and action tokenizer optimizer
        if not self.freeze_encoder:
            parameters = itertools.chain(
                self.encoder.parameters(),
                self.act_tok.parameters(),
            )
            self.encoder_opt = optimizer_class(parameters, lr=self.encoder_lr)
            
            # TACO optimizer
            if not self.no_taco:
                self.taco_opt = optimizer_class(
                    self.TACO.parameters(),
                    lr=self.encoder_lr
                )
        
        # Actor and critic optimizers
        self.actor_opt = optimizer_class(self.actor.parameters(), lr=self.lr)
        self.critic_opt = optimizer_class(self.critic.parameters(), lr=self.lr)
    
    def _load_checkpoint_components(
        self,
        checkpoint: Dict[str, Any],
        freeze_encoder: bool
    ):
        """
        Load TACO components from checkpoint.
        
        Args:
            checkpoint: Loaded checkpoint dictionary
            freeze_encoder: Whether encoder will be frozen
        """
        print("Checkpoint keys:", checkpoint.keys())
        
        # Load TACO module
        if 'taco' in checkpoint:
            self.TACO.load_checkpoint(checkpoint['taco'])
        else:
            utils.ColorPrint.yellow("! No TACO state found in checkpoint")
    
    def _freeze_components(self):
        """Freeze encoder, TACO, and action tokenizer."""
        self._frozen_fingerprints = {
            'encoder': self._get_model_fingerprint(self.encoder),
            'taco': self._get_model_fingerprint(self.TACO),
            'act_tok': self._get_model_fingerprint(self.act_tok)
        }
        
        # Set to eval mode
        self.encoder.eval()
        self.TACO.eval()
        self.act_tok.eval()
        
        # Disable gradients
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.TACO.parameters():
            param.requires_grad = False
        for param in self.act_tok.parameters():
            param.requires_grad = False
            
        utils.ColorPrint.blue("🔒 Encoder components frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder components."""
        if self._frozen_fingerprints is not None:
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.TACO.parameters():
                param.requires_grad = True
            for param in self.act_tok.parameters():
                param.requires_grad = True
            
            self.encoder.train()
            self.TACO.train()
            self.act_tok.train()
            
            super().unfreeze_encoder()
    
    def train(self, training: bool = True):
        """Set training mode for all components."""
        super().train(training)
        self.actor.train(training)
        self.critic.train(training)
        if not self.freeze_encoder:
            self.encoder.train(training)
            self.TACO.train(training)
    
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
            target_Q1, target_Q2 = self.critic_target(
                next_obs, next_action, self.act_tok
            )
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)
        
        Q1, Q2 = self.critic(obs, action, self.act_tok)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        
        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
        
        # Optimize
        if not self.freeze_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if not self.freeze_encoder:
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
        Q1, Q2 = self.critic(obs, action, self.act_tok)
        Q = torch.min(Q1, Q2)
        
        actor_loss = -Q.mean()
        
        # Optimize
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        
        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
        
        return metrics
    
    def update_taco(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        action_seq: torch.Tensor,
        next_obs: torch.Tensor,
        reward: torch.Tensor
    ) -> Dict[str, float]:
        """
        Update TACO contrastive learning module.
        
        Args:
            obs: Current observations
            action: Single actions
            action_seq: Action sequences for multistep
            next_obs: Next observations
            reward: Rewards
            
        Returns:
            Dictionary of metrics
        """
        metrics = dict()
        
        # Augment observations
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        
        # Encode states
        z = self.TACO.encode(obs)
        z_a = self.act_tok(action_seq)
        z_next = self.TACO.encode(next_obs, ema=False)
        
        # Project state-action pairs
        z_proj = self.TACO.project_sa(z, z_a)
        
        # Compute contrastive loss
        logits = self.TACO.compute_logits(z_proj, z_next)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        taco_loss = self.cross_entropy_loss(logits, labels)
        
        # Reward prediction loss
        reward_loss = torch.tensor(0.0)
        if self.reward:
            pred_reward = self.TACO.reward(torch.cat([z, z_a], dim=-1))
            reward_loss = F.mse_loss(pred_reward, reward)
        
        # Total loss
        loss = taco_loss + reward_loss
        
        # Optimize
        if not self.freeze_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)
            if not self.no_taco:
                self.taco_opt.zero_grad(set_to_none=True)
        
        loss.backward()
        
        if not self.freeze_encoder:
            self.encoder_opt.step()
            if not self.no_taco:
                self.taco_opt.step()
        
        if self.use_tb:
            metrics['taco_loss'] = taco_loss.item()
            metrics['reward_loss'] = reward_loss.item() if self.reward else 0.0
        
        return metrics
    
    def update(self, replay_iter, step: int) -> Dict[str, float]:
        """
        Perform a full update of the agent.
        
        Args:
            replay_iter: Iterator over replay buffer
            step: Current training step
            
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
        
        # Encode observations
        obs_en = self.encoder(obs.float())
        with torch.no_grad():
            next_obs_en = self.encoder(next_obs.float())
        
        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()
        
        # Update critic
        metrics.update(
            self.update_critic(obs_en, action, reward, discount, next_obs_en, step)
        )
        
        # Update actor
        metrics.update(self.update_actor(obs_en.detach(), step))
        
        # Soft update target critic
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )
        
        # Update TACO
        if self.no_taco:
            metrics['reward_loss'] = torch.tensor(0.0)
            metrics['taco_loss'] = torch.tensor(0.0)
        else:
            metrics.update(
                self.update_taco(obs, action, action_seq, r_next_obs, reward)
            )
        
        return metrics
