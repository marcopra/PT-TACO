"""
Base Agent Class

This is the foundational class for all RL agents. It provides:
- Common initialization patterns
- Pretrained model loading infrastructure
- Model freezing/unfreezing utilities
- Training mode management
- Abstract methods for agent-specific implementations
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Tuple
import utils


class BaseAgent(ABC):
    """
    Abstract base class for all reinforcement learning agents.
    
    This class defines the common interface and shared functionality
    for all agents in the framework.
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        device: torch.device,
        lr: float,
        use_tb: bool = False,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = False,
    ):
        """
        Initialize base agent.
        
        Args:
            obs_shape: Shape of observation space
            action_shape: Shape of action space
            device: Device to run computations on (cpu/cuda)
            lr: Learning rate
            use_tb: Whether to use tensorboard logging
            pretrained_path: Path to pretrained model checkpoint
            freeze_encoder: Whether to freeze encoder weights
        """
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.lr = lr
        self.use_tb = use_tb
        self.pretrained_path = pretrained_path
        self.freeze_encoder = freeze_encoder
        self.training = True
        
        # These will store fingerprints of frozen models
        self._frozen_fingerprints: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    
    def __getattribute__(self, name):
        return super().__getattribute__(name)
        
    @abstractmethod
    def build_networks(self):
        """
        Build all neural network components of the agent.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def build_optimizers(self):
        """
        Build all optimizers for the agent.
        Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def act(self, obs: torch.Tensor, step: int, eval_mode: bool) -> Any:
        """
        Select an action given an observation.
        
        Args:
            obs: Observation from environment
            step: Current training step
            eval_mode: Whether in evaluation mode
            
        Returns:
            Action to take in the environment
        """
        pass
    
    @abstractmethod
    def update(self, replay_iter, step: int) -> Dict[str, float]:
        """
        Update agent from a batch of experience.
        
        Args:
            replay_iter: Iterator over replay buffer
            step: Current training step
            
        Returns:
            Dictionary of metrics for logging
        """
        pass
    
    def train(self, training: bool = True):
        """
        Set the agent to training or evaluation mode.
        
        Args:
            training: Whether to set training mode (True) or eval mode (False)
        """
        self.training = training
        

    
    # @abstractmethod
    # def _load_components(
    #     self,
    #     checkpoint: Dict[str, Any],
    #     load_encoder: bool,
    #     load_actor: bool,
    #     load_critic: bool, 
    # ):
    #     """
    #     Load specific components from checkpoint.
    #     Must be implemented by subclasses.
        
    #     Args:
    #         checkpoint: Loaded checkpoint dictionary
    #         load_encoder: Whether to load encoder weights
    #         load_actor: Whether to load actor weights
    #         load_critic: Whether to load critic weights
    #     """
    #     pass
        
       

    def _freeze_encoder(self):
        """
        Freeze specified components after loading pretrained weights.
        Subclasses should override to specify which components to freeze.
        """
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        pass
    
    def _get_model_fingerprint(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Generate a unique fingerprint for model parameters.
        Useful for verifying frozen models haven't changed.
        
        Args:
            model: PyTorch module to fingerprint
            
        Returns:
            Dictionary mapping parameter names to cloned parameter tensors
        """
        return {name: param.data.clone() for name, param in model.named_parameters()}
    
    def _check_frozen_models(self):
        """
        Verify that frozen models have not been modified.
        Raises AssertionError if any frozen parameter has changed.
        """
        if self._frozen_fingerprints is None:
            raise ValueError(
                "No frozen fingerprints found. "
                "Did you load a pretrained model with freeze_encoder=True?"
            )
            
        for model_name, fingerprint in self._frozen_fingerprints.items():
            model = getattr(self, model_name)
            current_fingerprint = self._get_model_fingerprint(model)
            
            for param_name, stored_param in fingerprint.items():
                current_param = current_fingerprint[param_name]
                assert torch.all(torch.eq(current_param, stored_param)), (
                    f"Parameter {param_name} in {model_name} has changed "
                    f"when it should be frozen!"
                )
                
    def unfreeze_encoder(self):
        """
        Unfreeze all frozen components and restore gradient computation.
        Subclasses should override to specify which components to unfreeze.
        """
        if self._frozen_fingerprints is not None:
            self._frozen_fingerprints = None
            self.freeze_encoder = False
            utils.ColorPrint.blue("🔓 Components unfrozen")
            
    def save_checkpoint(self, filepath: str, step: int, **kwargs):
        """
        Save agent state to checkpoint file.
        
        Args:
            filepath: Where to save the checkpoint
            step: Current training step
            **kwargs: Additional items to save in checkpoint
        """
        checkpoint = {
            'step': step,
            **kwargs
        }
        torch.save(checkpoint, filepath)
        utils.ColorPrint.green(f"✓ Checkpoint saved to {filepath}")
