# Modular Agents Architecture

This directory contains a **modular and extensible** architecture for reinforcement learning agents. The design follows object-oriented principles with clear inheritance hierarchies, making it easy to understand, maintain, and extend.

## Architecture Overview

```
BaseAgent (abstract)
├── BaseTACOAgent (abstract)
│   ├── TACOAgent (image-based)
│   ├── TACOProprioAgent (state-based)
│   └── TACOTD3BCAgent (offline RL with behavior cloning)
└── DrQV2Agent (image-based without TACO)
```

## Core Components

### 1. `base_agent.py` - Foundation Class

**BaseAgent** is the abstract base class for all agents. It provides:

- **Common initialization patterns** for all agents
- **Pretrained model loading** infrastructure with checkpoint management
- **Model freezing/unfreezing** utilities for transfer learning
- **Training mode management** (train/eval switching)
- **Abstract methods** that force subclasses to implement specific functionality

**Key Methods:**
- `build_networks()` - Build all neural network components
- `build_optimizers()` - Create optimizers
- `act()` - Select actions given observations
- `update()` - Update agent from experience
- `load_pretrained()` - Load pretrained weights
- `_freeze_components()` - Freeze model parameters
- `_check_frozen_models()` - Verify frozen models haven't changed

### 2. `base_taco_agent.py` - TACO Base Class

**BaseTACOAgent** extends BaseAgent for agents using TACO (Temporal Action Contrastive Learning). It provides:

- **TACO module** initialization and management
- **Action tokenization** for embedding actions
- **Contrastive loss computation** for representation learning
- **Reward prediction** capabilities
- **Common update patterns** shared by all TACO-based agents

**Key Features:**
- Handles both frozen and trainable encoder scenarios
- Supports multiple optimizer types (Adam, SGD)
- Implements standard actor-critic updates with TACO augmentation
- Provides reward prediction as an auxiliary task

**Key Methods:**
- `update_critic()` - Twin Q-network critic update
- `update_actor()` - Policy gradient actor update
- `update_taco()` - Contrastive learning update
- `_freeze_components()` - Freeze encoder, TACO, and action tokenizer

### 3. `networks.py` - Reusable Network Components

Contains modular network architectures shared across agents:

**Encoders:**
- `ImageEncoder` - CNN for visual observations
- `ProprioceptiveEncoder` - MLP for state vectors

**Policy & Value Networks:**
- `Actor` - Stochastic policy network
- `Critic` - Twin Q-network critic

**TACO Components:**
- `TACOModule` - Complete contrastive learning module
- `RandomShiftsAug` - Data augmentation for images

**Benefits:**
- Code reuse across different agents
- Consistent network architectures
- Easy to modify and test individual components

## Concrete Agent Implementations

### 4. `taco_agent.py` - Image-Based TACO

**TACOAgent** for visual RL with convolutional encoder.

**Use Case:** Online RL with image observations
- Uses `ImageEncoder` for visual processing
- Full TACO contrastive learning
- Data augmentation with random shifts

### 5. `taco_proprio_agent.py` - State-Based TACO

**TACOProprioAgent** for low-dimensional state observations.

**Use Case:** Online RL with proprioceptive states
- Uses `ProprioceptiveEncoder` (MLP)
- Supports optional policy loading from checkpoint
- Configurable hidden layer architecture

### 6. `taco_td3bc_agent.py` - Offline RL with BC

**TACOTD3BCAgent** for offline reinforcement learning.

**Use Case:** Learning from fixed datasets without environment interaction
- Extends `TACOProprioAgent`
- Implements TD3+BC algorithm with behavior cloning regularization
- Custom actor/critic without trunks for better offline performance
- Alpha parameter controls BC regularization strength

**Key Features:**
- Behavior cloning loss prevents policy from deviating too far from data
- Works with proprioceptive states
- Suitable for learning from expert or sub-optimal datasets

### 7. `drqv2_agent.py` - Data-Regularized Q-Learning

**DrQV2Agent** for sample-efficient visual RL without TACO.

**Use Case:** Online RL with images, baseline comparison
- No contrastive learning (simpler than TACO)
- Strong data augmentation
- Works directly with raw actions (no tokenization)

## Design Principles

### 1. **Separation of Concerns**
Each class has a clear responsibility:
- Base classes define interfaces and common functionality
- Network modules are isolated and reusable
- Agent implementations focus on algorithm-specific logic

### 2. **DRY (Don't Repeat Yourself)**
Common code is in base classes:
- Pretrained model loading
- Freezing/unfreezing logic
- Training mode management
- Standard update patterns

### 3. **Open/Closed Principle**
Open for extension, closed for modification:
- New agents can extend base classes
- Override specific methods without changing base implementation
- Add new network types in `networks.py` without affecting agents

### 4. **Liskov Substitution Principle**
Any agent can be used where BaseAgent is expected:
- Consistent interface across all agents
- Polymorphic behavior
- Easy to swap agents in experiments

## Usage Examples

### Creating a New TACO Agent

```python
from modular_agents import TACOAgent

agent = TACOAgent(
    obs_shape=(3, 84, 84),  # RGB images
    action_shape=(6,),       # 6-DOF actions
    device=torch.device('cuda'),
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
    latent_a_dim='none',  # Auto-compute
    pretrained_path='checkpoints/pretrained.pt',
    freeze_encoder=False,
    no_taco=False
)
```

### Loading Pretrained Weights

```python
# Load and freeze encoder
agent.load_pretrained(
    'checkpoints/pretrained.pt',
    freeze_encoder=True
)

# Later, unfreeze for fine-tuning
agent.unfreeze_encoder()
```

### Training Loop

```python
for step in range(num_steps):
    # Act in environment
    action = agent.act(obs, step, eval_mode=False)
    next_obs, reward, done, _ = env.step(action)
    
    # Store in replay buffer
    replay_buffer.add(obs, action, reward, next_obs, done)
    
    # Update agent
    if step > initial_steps:
        metrics = agent.update(replay_iter, step)
        
        if step % log_freq == 0:
            logger.log_metrics(metrics, step)
```

## Extending the Architecture

### Adding a New Agent

1. **Choose base class:**
   - Extend `BaseAgent` for completely new algorithms
   - Extend `BaseTACOAgent` for TACO-based variants

2. **Implement required methods:**
   ```python
   class MyNewAgent(BaseTACOAgent):
       def build_networks(self):
           # Create your networks
           pass
       
       def update(self, replay_iter, step):
           # Implement your update logic
           pass
   ```

3. **Add to `__init__.py`:**
   ```python
   from .my_new_agent import MyNewAgent
   __all__ = [..., 'MyNewAgent']
   ```

### Adding New Network Components

Add to `networks.py`:

```python
class MyNewEncoder(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Your architecture
    
    def forward(self, x):
        # Your forward pass
        return encoded_x
```

## Comparison with Old Architecture

### Old Code Issues:
- ❌ Code duplication across files
- ❌ No clear inheritance structure
- ❌ Mixed responsibilities (networks + training logic)
- ❌ Hard to understand relationships between agents
- ❌ Difficult to add new agents without copy-paste

### New Modular Design:
- ✅ Clear class hierarchy with inheritance
- ✅ Shared code in base classes
- ✅ Separated network definitions
- ✅ Easy to understand and extend
- ✅ Follows SOLID principles
- ✅ Better maintainability and testability

## File Structure

```
modular_agents/
├── __init__.py                 # Package exports
├── README.md                   # This file
├── base_agent.py              # Abstract base for all agents
├── base_taco_agent.py         # Base for TACO agents
├── networks.py                # Reusable network components
├── taco_agent.py              # Image-based TACO
├── taco_proprio_agent.py      # State-based TACO
├── taco_td3bc_agent.py        # Offline RL with BC
└── drqv2_agent.py             # DrQV2 baseline
```

## Benefits Summary

1. **Maintainability:** Changes to common functionality only need to be made once
2. **Readability:** Clear hierarchy makes code easy to understand
3. **Extensibility:** New agents can be added with minimal code
4. **Testability:** Each component can be tested independently
5. **Reusability:** Networks and utilities can be shared across agents
6. **Type Safety:** Abstract methods ensure all agents implement required functionality

## Migration Guide

To use modular agents instead of old agents:

```python
# Old way
from agents.taco import TACOAgent

# New way
from modular_agents import TACOAgent

# API is compatible, just import from new location!
```

The interface is designed to be backward compatible, so existing code should work with minimal changes.
