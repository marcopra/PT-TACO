"""
TACO Agent (Image-based)

TACO agent for visual observations using convolutional encoder.
"""

import torch
from typing import Tuple, Optional, Any

from .base_taco_agent import BaseTACOAgent
from .components import ImageEncoder, Actor, Critic, TACO


class TACOAgent(BaseTACOAgent):
    """
    TACO agent for image-based observations.
    
    Uses a convolutional encoder for processing visual inputs and
    applies TACO contrastive learning for representation learning.
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
        curl: bool = False,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = False,
        no_taco: bool = False,
        optimizer_type: str = "adam"
    ):
        """
        Initialize TACO agent for image observations.
        
        Args:
            obs_shape: Shape of image observations (C, H, W)
            action_shape: Shape of action space
            device: Device for computation
            lr: Learning rate for actor and critic
            encoder_lr: Learning rate for encoder and TACO
            feature_dim: Dimension of feature representations
            hidden_dim: Dimension of hidden layers
            critic_target_tau: Soft update coefficient
            num_expl_steps: Number of exploration steps
            update_every_steps: Frequency of updates
            stddev_schedule: Schedule for action noise
            stddev_clip: Clipping value for action noise
            use_tb: Whether to use tensorboard
            reward: Whether to use reward prediction
            multistep: Number of action steps for TACO
            latent_a_dim: Dimension of latent action embeddings
            curl: Legacy parameter for compatibility
            pretrained_path: Path to pretrained checkpoint
            freeze_encoder: Whether to freeze encoder
            no_taco: Whether to disable TACO loss
            optimizer_type: Type of optimizer
        """
        super().__init__(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            lr=lr,
            encoder_lr=encoder_lr,
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
        
        self.curl = curl  # For compatibility
        
        # Build networks
        self.build_networks()
        
        # Build optimizers
        self.build_optimizers()
        
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
        
        # Critic and target critic
        self.critic = Critic(
            self.encoder.repr_dim,
            self.latent_a_dim,
            self.feature_dim,
            self.hidden_dim
        ).to(self.device)
        
        self.critic_target = Critic(
            self.encoder.repr_dim,
            self.latent_a_dim,
            self.feature_dim,
            self.hidden_dim
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # TACO module
        self.TACO = TACO(
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
