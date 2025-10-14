# Configuration Structure

This directory contains the hierarchical configuration system for PT-TACO experiments.

## Structure

```
cfgs/
├── config_gym_new.yaml          # Base configuration (entry point)
├── agent/                       # Agent-specific configurations
│   ├── drqv2_new.yaml          # DrQV2 base config
│   ├── taco.yaml               # TACO base config
│   └── ...
├── task/                        # Task-specific configurations
│   ├── PointMaze_MediumDense.yaml
│   ├── walker_walk.yaml
│   └── ...
├── experiment/                  # Experiment configurations
│   ├── drqv2_pointmaze_base.yaml
│   ├── drqv2_pointmaze_pretrained.yaml
│   └── drqv2_walker_walk.yaml
└── wrapper/                     # Wrapper configurations
    └── taco_wrapper.yaml
```

## Usage

### 1. Using Base Configuration

Run with default settings from `config_gym_new.yaml`:

```bash
python train_gym.py
```

### 2. Using Experiment Configuration

Run with a predefined experiment configuration:

```bash
python train_gym.py experiment=drqv2_pointmaze_base
```

This will:
- Load `config_gym_new.yaml` as base
- Override with settings from `experiment/drqv2_pointmaze_base.yaml`
- Use the agent and task specified in the experiment config

### 3. Override Specific Parameters

Override any parameter from command line:

```bash
# Override learning rate
python train_gym.py experiment=drqv2_pointmaze_base lr=5e-4

# Override multiple parameters
python train_gym.py experiment=drqv2_pointmaze_base lr=5e-4 batch_size=128 seed=42

# Use different agent
python train_gym.py experiment=drqv2_pointmaze_base agent@_global_=taco

# Disable wandb
python train_gym.py experiment=drqv2_pointmaze_base use_wandb=false
```

### 4. Using Pretrained Encoder

```bash
python train_gym.py experiment=drqv2_pointmaze_pretrained \
    pretrained_path=/path/to/encoder.pt \
    freeze_encoder=true
```

## Hierarchy

The configuration system follows this priority (highest to lowest):

1. **Command line overrides** (highest priority)
2. **Experiment config** (`experiment/*.yaml`)
3. **Agent/Task config** (`agent/*.yaml`, `task/*.yaml`)
4. **Base config** (`config_gym_new.yaml`) (lowest priority)

## Creating New Experiments

To create a new experiment:

1. Create a new file in `cfgs/experiment/`:

```yaml
# @package _global_

# My custom experiment
defaults:
  - override /task: my_task
  - override /agent: my_agent

# Experiment name
exp_name: my_experiment_name

# Override any parameters
lr: 1e-3
batch_size: 512
num_train_frames: 2000000

# Wandb settings
wandb_run_name: ${now:%Y.%m.%d}_${now:%H%M}_my_experiment
wandb_tag: my_tag
```

2. Run it:

```bash
python train_gym.py experiment=my_experiment_name
```

## Example Experiments

### DrQV2 on PointMaze (Base)
```bash
python train_gym.py experiment=drqv2_pointmaze_base
```

### DrQV2 on PointMaze (Pretrained)
```bash
python train_gym.py experiment=drqv2_pointmaze_pretrained \
    pretrained_path=/path/to/checkpoint/encoder.pt
```

### DrQV2 on Walker Walk (Vision)
```bash
python train_gym.py experiment=drqv2_walker_walk
```

## Tips

- Use `python train_gym.py --help` to see all available options
- Use `python train_gym.py --cfg job` to print the final merged configuration
- Use `experiment=` to load predefined experiment configurations
- All parameters can be overridden from command line
