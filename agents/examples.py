"""
Example Usage of Modular Agents

This script demonstrates how to use the modular agent architecture.
"""

import torch
from agents import (
    TACOAgent,
    TACOProprioAgent,
    TACOTD3BCAgent,
    DrQV2Agent
)


def example_taco_image_agent():
    """Example: Create a TACO agent for image-based tasks."""
    print("=" * 60)
    print("Example 1: TACO Agent (Image-Based)")
    print("=" * 60)
    
    agent = TACOAgent(
        obs_shape=(3, 84, 84),  # RGB images
        action_shape=(6,),       # 6-DOF action space
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        lr=1e-4,
        encoder_lr=1e-4,
        feature_dim=50,
        hidden_dim=1024,
        critic_target_tau=0.01,
        num_expl_steps=2000,
        update_every_steps=2,
        stddev_schedule='linear(1.0,0.1,100000)',
        stddev_clip=0.3,
        use_tb=True,
        reward=True,
        multistep=3,
        latent_a_dim='none',  # Auto-computed
        pretrained_path=None,
        freeze_encoder=False,
        no_taco=False
    )
    
    print(f"✓ Created TACO agent")
    print(f"  - Encoder repr_dim: {agent.encoder.repr_dim}")
    print(f"  - Latent action dim: {agent.latent_a_dim}")
    print(f"  - Freeze encoder: {agent.freeze_encoder}")
    print()
    
    return agent


def example_taco_proprio_agent():
    """Example: Create a TACO agent for proprioceptive states."""
    print("=" * 60)
    print("Example 2: TACO Agent (Proprioceptive States)")
    print("=" * 60)
    
    agent = TACOProprioAgent(
        obs_shape=(39,),  # 39-dim state vector
        action_shape=(8,),  # 8-dim action space
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        lr=1e-4,
        encoder_lr=1e-4,
        repr_dim=512,  # MLP encoder output
        encoder_hidden_dim=[1024, 1024],  # Two hidden layers
        feature_dim=50,
        hidden_dim=1024,
        critic_target_tau=0.01,
        num_expl_steps=2000,
        update_every_steps=2,
        stddev_schedule='linear(1.0,0.1,100000)',
        stddev_clip=0.3,
        use_tb=True,
        reward=True,
        multistep=3,
        latent_a_dim='none',
        pretrained_path=None,
        freeze_encoder=False,
        no_taco=False
    )
    
    print(f"✓ Created TACO Proprio agent")
    print(f"  - Encoder repr_dim: {agent.encoder.repr_dim}")
    print(f"  - Encoder hidden dims: {agent.encoder_hidden_dim}")
    print(f"  - Latent action dim: {agent.latent_a_dim}")
    print()
    
    return agent


def example_td3bc_agent():
    """Example: Create a TD3BC agent for offline RL."""
    print("=" * 60)
    print("Example 3: TD3BC Agent (Offline RL)")
    print("=" * 60)
    
    agent = TACOTD3BCAgent(
        name='td3bc_offline',
        obs_shape=(39,),
        action_shape=(8,),
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        lr=3e-4,
        encoder_lr=3e-4,
        repr_dim=512,
        encoder_hidden_dim=[1024, 1024],
        feature_dim=50,
        hidden_dim=1024,
        critic_target_tau=0.005,
        num_expl_steps=0,  # No exploration in offline RL
        update_every_steps=1,
        stddev_schedule='linear(1.0,0.1,100000)',
        nstep=3,
        batch_size=256,
        stddev_clip=0.3,
        use_tb=True,
        reward=True,
        multistep=3,
        latent_a_dim='none',
        alpha=2.5,  # BC regularization strength
        pretrained_path=None,
        freeze_encoder=False,
        no_taco=False
    )
    
    print(f"✓ Created TD3BC agent")
    print(f"  - Alpha (BC regularization): {agent.alpha}")
    print(f"  - Actor type: {type(agent.actor).__name__}")
    print(f"  - Critic type: {type(agent.critic).__name__}")
    print()
    
    return agent


def example_drqv2_agent():
    """Example: Create a DrQV2 agent."""
    print("=" * 60)
    print("Example 4: DrQV2 Agent (Baseline)")
    print("=" * 60)
    
    agent = DrQV2Agent(
        obs_shape=(3, 84, 84),
        action_shape=(6,),
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        lr=1e-4,
        feature_dim=50,
        hidden_dim=1024,
        critic_target_tau=0.01,
        num_expl_steps=2000,
        update_every_steps=2,
        stddev_schedule='linear(1.0,0.1,100000)',
        stddev_clip=0.3,
        use_tb=True,
        pretrained_path=None,
        freeze_encoder=False
    )
    
    print(f"✓ Created DrQV2 agent")
    print(f"  - Encoder repr_dim: {agent.encoder.repr_dim}")
    print(f"  - No TACO module (baseline)")
    print()
    
    return agent


def example_pretrained_loading():
    """Example: Load pretrained weights and freeze encoder."""
    print("=" * 60)
    print("Example 5: Loading Pretrained Weights")
    print("=" * 60)
    
    # Note: This assumes you have a pretrained checkpoint
    # For demonstration, we'll show the API without actual file
    
    print("Creating agent...")
    agent = TACOAgent(
        obs_shape=(3, 84, 84),
        action_shape=(6,),
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        lr=1e-4,
        encoder_lr=1e-4,
        feature_dim=50,
        hidden_dim=1024,
        critic_target_tau=0.01,
        num_expl_steps=2000,
        update_every_steps=2,
        stddev_schedule='linear(1.0,0.1,100000)',
        stddev_clip=0.3,
        use_tb=True,
        reward=True,
        multistep=3,
        latent_a_dim='none',
        pretrained_path=None,  # Will load manually
        freeze_encoder=False
    )
    
    print("\nLoading pretrained weights (example API):")
    print("  agent.load_pretrained('path/to/checkpoint.pt', freeze_encoder=True)")
    print("  # This would load and freeze the encoder")
    
    print("\nUnfreezing encoder for fine-tuning:")
    print("  agent.unfreeze_encoder()")
    print("  # Now encoder can be updated again")
    
    print("\nChecking if frozen models have changed:")
    print("  agent._check_frozen_models()")
    print("  # Raises error if frozen parameters were modified")
    print()


def example_action_selection():
    """Example: Using agents for action selection."""
    print("=" * 60)
    print("Example 6: Action Selection")
    print("=" * 60)
    
    agent = TACOProprioAgent(
        obs_shape=(39,),
        action_shape=(8,),
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        lr=1e-4,
        encoder_lr=1e-4,
        repr_dim=512,
        encoder_hidden_dim=[1024, 1024],
        feature_dim=50,
        hidden_dim=1024,
        critic_target_tau=0.01,
        num_expl_steps=2000,
        update_every_steps=2,
        stddev_schedule='linear(1.0,0.1,100000)',
        stddev_clip=0.3,
        use_tb=True,
        reward=True,
        multistep=3,
        latent_a_dim='none',
        pretrained_path=None,
        freeze_encoder=False,
        no_taco=False
    )
    
    # Simulate observation
    obs = torch.randn(39)
    
    # Training mode action (with exploration)
    print("Training mode (with exploration):")
    action_train = agent.act(obs, step=1000, eval_mode=False)
    print(f"  Action shape: {action_train.shape}")
    print(f"  Action: {action_train}")
    
    # Evaluation mode action (deterministic)
    print("\nEvaluation mode (deterministic):")
    action_eval = agent.act(obs, step=1000, eval_mode=True)
    print(f"  Action shape: {action_eval.shape}")
    print(f"  Action: {action_eval}")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "MODULAR AGENTS - USAGE EXAMPLES" + " " * 16 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Run examples
    example_taco_image_agent()
    example_taco_proprio_agent()
    example_td3bc_agent()
    example_drqv2_agent()
    example_pretrained_loading()
    example_action_selection()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Check modular_agents/README.md for detailed documentation")
    print("  2. Use these agents in your training scripts")
    print("  3. Extend base classes to create new agent variants")
    print()


if __name__ == '__main__':
    main()
