"""
TACO Agent (Proprioceptive States)

TACO agent for low-dimensional state observations using MLP encoder.
"""

import torch
from typing import Tuple, Optional, Any, Dict, List

from .base_taco_agent import BaseTACOAgent
from .components import ProprioceptiveEncoder, Actor, Critic, TACO


class TACOProprioAgent(BaseTACOAgent):
    """
    TACO agent for proprioceptive (low-dimensional) state observations.
    
    Uses an MLP encoder for processing state vectors and applies TACO
    contrastive learning for representation learning.
    """
    
    def __init__(
        self,
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
        Initialize TACO agent for proprioceptive observations.
        
        Args:
            obs_shape: Shape of state observations (must be 1D)
            action_shape: Shape of action space
            device: Device for computation
            lr: Learning rate for actor and critic
            encoder_lr: Learning rate for encoder and TACO
            repr_dim: Dimension of encoder output
            encoder_hidden_dim: List of hidden layer sizes for encoder
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
            pretrained_path: Path to pretrained checkpoint
            freeze_encoder: Whether to freeze encoder
            no_taco: Whether to disable TACO loss
            optimizer_type: Type of optimizer
        """
        # Store encoder-specific parameters before calling super().__init__
        self.repr_dim = repr_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        
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
        # Proprioceptive encoder (MLP)
        self.encoder = ProprioceptiveEncoder(
            self.obs_shape,
            self.repr_dim,
            self.encoder_hidden_dim
        ).to(self.device)
        
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
    
    def _load_checkpoint_components(
        self,
        checkpoint: Dict[str, Any],
        freeze_encoder: bool
    ):
        """
        Load TACO components from checkpoint, with optional policy loading.
        
        Args:
            checkpoint: Loaded checkpoint dictionary
            freeze_encoder: Whether encoder will be frozen
        """
        print("Checkpoint keys:", checkpoint.keys())
        
        # Load TACO module
        if 'taco' in checkpoint:
            self.TACO.load_checkpoint(checkpoint['taco'])
        else:
            import utils
            utils.ColorPrint.yellow("! No TACO state found in checkpoint")
    
    def load_pretrained(
        self,
        model_path: str,
        map_location: Optional[torch.device] = None,
        freeze_encoder: bool = False,
        policy: bool = False
    ) -> Dict[str, Any]:
        """
        Load pretrained model with optional policy loading.
        
        Args:
            model_path: Path to checkpoint
            map_location: Device mapping
            freeze_encoder: Whether to freeze encoder
            policy: Whether to load actor weights
            
        Returns:
            Dictionary of saved arguments
        """
        if map_location is None:
            map_location = self.device
            
        import utils
        utils.ColorPrint.blue(f"Loading pretrained model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=map_location, weights_only=False)
        
        # Load TACO components
        self._load_checkpoint_components(checkpoint, freeze_encoder)
        
        # Load actor if requested
        if policy and 'actor' in checkpoint:
            try:
                self.actor.load_state_dict(checkpoint['actor'], strict=True)
                utils.ColorPrint.green("✓ Loaded actor from checkpoint")
            except RuntimeError as e:
                utils.ColorPrint.red(f"✗ Failed to load actor: {e}")
        elif policy:
            utils.ColorPrint.yellow("! No actor state found in checkpoint")
        else:
            utils.ColorPrint.blue("Skipping actor loading as policy=False")
        
        # Freeze if requested
        if freeze_encoder:
            self._freeze_components()
        
        utils.ColorPrint.green("✓ Pretrained model loading completed")
        return checkpoint.get('args', {})
