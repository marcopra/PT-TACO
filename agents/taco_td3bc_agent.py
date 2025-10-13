"""
TACO TD3BC Agent

TD3+BC agent with TACO representations for offline reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Any, Dict, List
import utils

from .taco_proprio_agent import TACOProprioAgent


class ActorTD3BC(nn.Module):
    """
    Actor network for TD3BC (without trunk, operates on raw observations).
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        """
        Args:
            obs_dim: Dimension of observation features
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.apply(utils.weight_init)
    
    def forward(self, obs: torch.Tensor, std: float) -> utils.TruncatedNormal:
        """
        Compute action distribution.
        
        Args:
            obs: Encoded observation tensor
            std: Standard deviation for exploration
            
        Returns:
            Truncated normal distribution over actions
        """
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        std_tensor = torch.ones_like(mu) * std
        
        dist = utils.TruncatedNormal(mu, std_tensor)
        return dist


class CriticTD3BC(nn.Module):
    """
    Twin Q-network critic for TD3BC (without trunk).
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        """
        Args:
            obs_dim: Dimension of observation features
            action_dim: Dimension of action space
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        self.q1_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(utils.weight_init)
    
    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q-values for observation-action pairs.
        
        Args:
            obs: Observation tensor
            action: Action tensor (already tokenized if needed)
            
        Returns:
            Tuple of (Q1, Q2) values
        """
        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.q1_net(obs_action)
        q2 = self.q2_net(obs_action)
        
        return q1, q2


class TACOTD3BCAgent(TACOProprioAgent):
    """
    TD3BC agent with TACO representations.
    
    Implements TD3+BC algorithm for offline RL, where the actor
    is regularized to stay close to the behavior policy.
    """
    
    def __init__(
        self,
        name: str,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        device: torch.device,
        lr: float,
        encoder_lr: float,
        repr_dim: int,
        encoder_hidden_dim: List[int],
        feature_dim: int,
        hidden_dim: int,
        critic_target_tau: float,
        num_expl_steps: int,
        update_every_steps: int,
        stddev_schedule: str,
        nstep: int,
        batch_size: int,
        stddev_clip: float,
        use_tb: bool,
        reward: bool,
        multistep: int,
        latent_a_dim: Any,
        alpha: float,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = False,
        no_taco: bool = False,
        optimizer_type: str = "adam",
        has_next_action: bool = False
    ):
        """
        Initialize TD3BC agent.
        
        Args:
            name: Agent name
            obs_shape: Shape of state observations
            action_shape: Shape of action space
            device: Device for computation
            lr: Learning rate for actor and critic
            encoder_lr: Learning rate for encoder
            repr_dim: Dimension of encoder output
            encoder_hidden_dim: Hidden layer sizes for encoder
            feature_dim: Dimension of features
            hidden_dim: Dimension of hidden layers
            critic_target_tau: Soft update coefficient
            num_expl_steps: Number of exploration steps
            update_every_steps: Update frequency
            stddev_schedule: Action noise schedule
            nstep: N-step returns (unused, for compatibility)
            batch_size: Batch size (unused, for compatibility)
            stddev_clip: Clipping for action noise
            use_tb: Whether to use tensorboard
            reward: Whether to use reward prediction
            multistep: Number of action steps for TACO
            latent_a_dim: Latent action dimension
            alpha: TD3BC regularization coefficient
            pretrained_path: Path to pretrained checkpoint
            freeze_encoder: Whether to freeze encoder
            no_taco: Whether to disable TACO
            optimizer_type: Optimizer type
            has_next_action: Whether dataset has next actions (unused)
        """
        # Initialize parent class
        super().__init__(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            lr=lr,
            encoder_lr=encoder_lr,
            repr_dim=repr_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            critic_target_tau=critic_target_tau,
            num_expl_steps=num_expl_steps,
            update_every_steps=update_every_steps,
            stddev_schedule=stddev_schedule,
            stddev_clip=stddev_clip,
            use_tb=use_tb,
            reward=reward,
            multistep=multistep,
            latent_a_dim=latent_a_dim,
            pretrained_path=pretrained_path,
            freeze_encoder=freeze_encoder,
            no_taco=no_taco,
            optimizer_type=optimizer_type
        )
        
        # TD3BC specific parameters
        self.name = name
        self.alpha = alpha
        self.action_dim = action_shape[0]
    
    def build_networks(self):
        """Build networks for TD3BC agent."""
        # First build encoder and TACO from parent
        from .networks import ProprioceptiveEncoder, TACOModule
        
        self.encoder = ProprioceptiveEncoder(
            self.obs_shape,
            self.repr_dim,
            self.encoder_hidden_dim
        ).to(self.device)
        
        # Use TD3BC-specific actor (no trunk)
        self.actor = ActorTD3BC(
            self.encoder.repr_dim,
            self.action_dim,
            self.hidden_dim
        ).to(self.device)
        
        # Use TD3BC-specific critic (no trunk)
        self.critic = CriticTD3BC(
            self.encoder.repr_dim,
            self.action_dim,
            self.hidden_dim
        ).to(self.device)
        
        self.critic_target = CriticTD3BC(
            self.encoder.repr_dim,
            self.action_dim,
            self.hidden_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # TACO module
        self.TACO = TACOModule(
            self.encoder.repr_dim,
            self.feature_dim,
            self.action_shape,
            self.latent_a_dim,
            self.hidden_dim,
            self.act_tok,
            self.encoder,
            self.multistep,
            self.device
        ).to(self.device)
    
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
        Update critic networks (TD3BC version).
        
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
            
            # TD3BC: Critic doesn't use action tokenizer
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
        
        # Optimize
        if not self.freeze_encoder:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if not self.freeze_encoder:
            self.encoder_opt.step()
        
        return metrics
    
    def update_actor(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        step: int
    ) -> Dict[str, float]:
        """
        Update actor with TD3BC behavior cloning regularization.
        
        Args:
            obs: Observations
            action: Behavior policy actions
            step: Current step
            
        Returns:
            Dictionary of metrics
        """
        metrics = dict()
        
        stddev = utils.schedule(self.stddev_schedule, step)
        policy = self.actor(obs, stddev)
        
        # Compute Q-values (with action tokenizer for TACO compatibility)
        Q1, Q2 = self.critic(obs, policy.sample(clip=self.stddev_clip))
        Q = torch.min(Q1, Q2)
        
        # TD3BC loss: weighted combination of Q-maximization and BC
        lmbda = self.alpha / Q.abs().mean().detach()
        actor_loss = -lmbda * Q.mean() + F.mse_loss(policy.mean, action)
        
        # Optimize
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        
        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_ent'] = policy.entropy().sum(dim=-1).mean().item()
        
        return metrics
    
    def update(self, replay_iter, step: int) -> Dict[str, float]:
        """
        Full update for TD3BC agent.
        
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
        
        # Encode observations
        obs_en = obs.float()
        next_obs_en = next_obs.float()
        obs_en = self.encoder(obs_en)
        with torch.no_grad():
            next_obs_en = self.encoder(next_obs_en)
        
        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()
        
        # Update critic
        metrics.update(
            self.update_critic(obs_en, action, reward, discount, next_obs_en, step)
        )
        
        # Update actor (with behavior cloning)
        metrics.update(self.update_actor(obs_en.detach(), action, step))
        
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
