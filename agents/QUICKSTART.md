# Quick Start Guide - Modular Agents

## Installation

No installation needed! The modular agents are already in your project.

```bash
cd /home/mprattico/Pretrain-TACO
```

## Quick Test

Run the examples to verify everything works:

```bash
python -m modular_agents.examples
```

This will instantiate all agents and show you how to use them.

## 5-Minute Tutorial

### 1. Import an Agent

```python
from modular_agents import TACOAgent
import torch
```

### 2. Create Agent

```python
agent = TACOAgent(
    obs_shape=(3, 84, 84),      # RGB images
    action_shape=(6,),           # 6D actions
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
    latent_a_dim='none',
)
```

### 3. Use Agent

```python
# Select action
obs = env.reset()
action = agent.act(obs, step=0, eval_mode=False)

# Take step
next_obs, reward, done, _ = env.step(action)

# Update agent
replay_buffer.add(obs, action, reward, next_obs, done)
metrics = agent.update(replay_iter, step=0)
```

### 4. Load Pretrained Weights

```python
# Load and freeze encoder
agent.load_pretrained(
    'path/to/checkpoint.pt',
    freeze_encoder=True
)

# Later, unfreeze for fine-tuning
agent.unfreeze_encoder()
```

## Available Agents

### For Image Observations

#### TACO Agent (with contrastive learning)
```python
from modular_agents import TACOAgent

agent = TACOAgent(
    obs_shape=(3, 84, 84),
    action_shape=(6,),
    # ... other params
)
```

#### DrQV2 (baseline, no contrastive learning)
```python
from modular_agents import DrQV2Agent

agent = DrQV2Agent(
    obs_shape=(3, 84, 84),
    action_shape=(6,),
    # ... other params
)
```

### For Proprioceptive States

#### TACO Proprio Agent
```python
from modular_agents import TACOProprioAgent

agent = TACOProprioAgent(
    obs_shape=(39,),  # State vector
    action_shape=(8,),
    repr_dim=512,
    encoder_hidden_dim=[1024, 1024],
    # ... other params
)
```

#### TD3BC (offline RL)
```python
from modular_agents import TACOTD3BCAgent

agent = TACOTD3BCAgent(
    name='my_td3bc_agent',
    obs_shape=(39,),
    action_shape=(8,),
    alpha=2.5,  # BC regularization
    # ... other params
)
```

## Common Tasks

### Training from Scratch

```python
from modular_agents import TACOAgent

# Create agent
agent = TACOAgent(...)

# Training loop
for step in range(num_steps):
    # Collect experience
    action = agent.act(obs, step, eval_mode=False)
    next_obs, reward, done, _ = env.step(action)
    replay_buffer.add(obs, action, reward, next_obs, done)
    
    # Update
    if step > warmup_steps:
        metrics = agent.update(replay_iter, step)
        logger.log(metrics, step)
    
    obs = next_obs if not done else env.reset()
```

### Transfer Learning

```python
# 1. Pretrain on task A
agent_a = TACOAgent(...)
# ... train agent_a ...
agent_a.save_checkpoint('pretrained.pt', step=100000)

# 2. Transfer to task B with frozen encoder
agent_b = TACOAgent(...)
agent_b.load_pretrained('pretrained.pt', freeze_encoder=True)
# ... train agent_b (only actor/critic update) ...

# 3. Fine-tune everything
agent_b.unfreeze_encoder()
# ... continue training with encoder updates ...
```

### Offline RL

```python
from modular_agents import TACOTD3BCAgent

# Create agent
agent = TACOTD3BCAgent(
    name='offline_agent',
    obs_shape=(39,),
    action_shape=(8,),
    alpha=2.5,  # Higher = more BC, lower = more RL
    # ... other params
)

# Training loop (no environment interaction!)
for step in range(num_steps):
    # Sample from fixed dataset
    batch = dataset.sample(batch_size)
    
    # Update
    metrics = agent.update(batch, step)
    logger.log(metrics, step)
```

## Extending the Architecture

### Create a Custom Agent

```python
from modular_agents.base_taco_agent import BaseTACOAgent
from modular_agents.networks import ImageEncoder, Actor, Critic, TACO

class MyCustomAgent(BaseTACOAgent):
    """My custom TACO variant."""
    
    def build_networks(self):
        """Build custom networks."""
        # Use existing components
        self.encoder = ImageEncoder(self.obs_shape, self.feature_dim)
        self.actor = Actor(...)
        self.critic = Critic(...)
        self.TACO = TACO(...)
        
        # Or create custom ones
        self.my_custom_module = MyCustomModule(...)
    
    def update(self, replay_iter, step):
        """Custom update logic."""
        # Call parent update
        metrics = super().update(replay_iter, step)
        
        # Add custom updates
        custom_metrics = self.update_custom_module(...)
        metrics.update(custom_metrics)
        
        return metrics
```

### Add to Package

Edit `modular_agents/__init__.py`:
```python
from .my_custom_agent import MyCustomAgent

__all__ = [
    # ... existing agents ...
    'MyCustomAgent',
]
```

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the project root
cd /home/mprattico/Pretrain-TACO

# Import should work
python -c "from modular_agents import TACOAgent; print('OK')"
```

### CUDA Out of Memory
```python
# Use smaller batch size
agent = TACOAgent(..., batch_size=128)  # instead of 256

# Or smaller hidden dimensions
agent = TACOAgent(..., hidden_dim=512)  # instead of 1024
```

### Encoder Not Freezing
```python
# Verify freezing worked
agent.load_pretrained('checkpoint.pt', freeze_encoder=True)

# Check if parameters require gradients
for name, param in agent.encoder.named_parameters():
    print(f"{name}: requires_grad={param.requires_grad}")
    # Should print: requires_grad=False

# Use check method
agent._check_frozen_models()  # Raises error if modified
```

## Documentation

- **README.md** - Comprehensive architecture documentation
- **IMPLEMENTATION_SUMMARY.md** - What was created and why
- **ARCHITECTURE_COMPARISON.md** - Old vs new comparison
- **examples.py** - Working code examples

## Next Steps

1. ✅ Run examples: `python -m modular_agents.examples`
2. ✅ Read README.md for detailed documentation
3. ✅ Try creating an agent in your own script
4. ✅ Customize for your specific task
5. ✅ Extend with your own variants

## Help

If you need help:
1. Check the examples in `examples.py`
2. Read the docstrings in the code
3. Look at the README.md
4. Review the implementation of existing agents

## Summary

The modular agents provide:
- ✅ Clean, professional architecture
- ✅ Easy to use and extend
- ✅ Well-documented
- ✅ Production-ready
- ✅ No changes to existing code

**You're ready to go! Start with `python -m modular_agents.examples`**
