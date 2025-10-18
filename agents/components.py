"""
Common Network Architectures

This module contains reusable network components that are shared
across different agents.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from typing import List, Tuple


class RandomShiftsAug(nn.Module):
    """
    Data augmentation module that applies random spatial shifts to images.
    Used for visual observation augmentation.
    """
    
    def __init__(self, pad: int = 4):
        """
        Args:
            pad: Padding size for shifts
        """
        super().__init__()
        self.pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random shifts to input images.
        
        Args:
            x: Input tensor of shape (N, C, H, W)
            
        Returns:
            Shifted tensor of same shape
        """
        n, c, h, w = x.size()
        assert h == w, "Input images must be square"
        
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps,
            1.0 - eps,
            h + 2 * self.pad,
            device=x.device,
            dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0,
            2 * self.pad + 1,
            size=(n, 1, 1, 2),
            device=x.device,
            dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


class ImageEncoder(nn.Module):
    """
    Convolutional encoder for image observations.
    Used in visual RL tasks.
    """
    
    def __init__(self, obs_shape: Tuple[int, ...], feature_dim: int = 50):
        """
        Args:
            obs_shape: Shape of observation (C, H, W)
            feature_dim: Dimension of output features (not used, for compatibility)
        """
        super().__init__()
        assert len(obs_shape) == 3, "Image encoder expects 3D observations (C, H, W)"
        
        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU()
        )
        
        # Compute representation dimension dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            dummy_output = self.convnet(dummy_input)
            self.repr_dim = dummy_output.view(dummy_output.shape[0], -1).shape[1]
        
        self.apply(utils.weight_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode image observation.
        
        Args:
            obs: Image tensor of shape (N, C, H, W)
            
        Returns:
            Encoded features of shape (N, repr_dim)
        """
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class ProprioceptiveEncoder(nn.Module):
    """
    MLP encoder for proprioceptive state observations.
    Used when working with low-dimensional state vectors.
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        repr_dim: int,
        hidden_dim: List[int] = [1024, 1024]
    ):
        """
        Args:
            obs_shape: Shape of observation (must be 1D)
            repr_dim: Output representation dimension
            hidden_dim: List of hidden layer sizes
        """
        super().__init__()
        assert len(obs_shape) == 1, (
            f"Proprioceptive encoder expects 1D observations, got {obs_shape}"
        )
        
        self.encoder = utils.create_mlp(
            input_dim=obs_shape[0],
            output_dim=repr_dim,
            hidden_sizes=hidden_dim,
            output_activation="relu"
        )
        self.repr_dim = repr_dim
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encode proprioceptive observation.
        
        Args:
            obs: State tensor of shape (N, obs_dim)
            
        Returns:
            Encoded features of shape (N, repr_dim)
        """
        h = self.encoder(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    """
    Actor network that outputs a stochastic policy.
    Maps encoded observations to action distributions.
    """
    
    def __init__(
        self,
        repr_dim: int,
        action_dim: int,
        feature_dim: int,
        hidden_dim: int
    ):
        """
        Args:
            repr_dim: Dimension of encoded observations
            action_shape: Shape of action space
            feature_dim: Dimension of feature layer
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh()
        )
        
        self.policy = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
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
        h = self.trunk(obs)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std_tensor = torch.ones_like(mu) * std
        
        dist = utils.TruncatedNormal(mu, std_tensor)
        return dist


class Critic(nn.Module):
    """
    Twin Q-network critic.
    Estimates state-action values using two Q-functions.
    """
    
    def __init__(
        self,
        repr_dim: int,
        action_dim: int,
        feature_dim: int,
        hidden_dim: int
    ):
        """
        Args:
            repr_dim: Dimension of encoded observations
            action_dim: Dimension of actions (or latent actions)
            feature_dim: Dimension of feature layer
            hidden_dim: Dimension of hidden layers
        """
        super().__init__()
        
        self.trunk = nn.Sequential(
            nn.Linear(repr_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh()
        )
        
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        self.apply(utils.weight_init)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        act_tok: nn.Module = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q-values for observation-action pairs.
        
        Args:
            obs: Encoded observation tensor
            action: Action tensor
            act_tok: Optional action tokenizer for TACO agents
            
        Returns:
            Tuple of (Q1, Q2) values
        """
        # Apply action tokenizer if provided
        if act_tok is not None:
            action = act_tok(action)
            
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        
        return q1, q2

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
