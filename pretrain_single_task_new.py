"""
The script pretrain on dataset coming from a single task, it means that there is no target task from which we want to test. 
"""
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import torch
import utils
import wandb
import json
from pathlib import Path
from pretraining_utils import *



def make_wrapped_agent(obs_shape, action_shape, agent_cfg, wrapper_cfg):
    # Handle agent configuration from global config
    agent_cfg.obs_shape = obs_shape
    agent_cfg.action_shape = action_shape

    if wrapper_cfg is not None:
        print(f"Using wrapper: {wrapper_cfg._target_}")
        # Instantiate base agent first
        base_algorithm = hydra.utils.instantiate(agent_cfg)
        
        # Pass required parameters to wrapper
        return hydra.utils.instantiate(wrapper_cfg, base_algorithm=base_algorithm)

    return hydra.utils.instantiate(agent_cfg)


@hydra.main(config_path='cfgs', config_name='config_pretrain_ST_new', version_base="1.1")
def main(cfg: DictConfig):
    print("Pretraining Starting:")
    
    # Set device and seed
    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)
    
    # Compatibility checks
    assert cfg.multistep == cfg.nstep, f"Don't know the difference between nstep and multistep, set them to the same value"

    # Parse checkpoint steps from string to list of integers
    checkpoint_steps = [int(step) for step in cfg.checkpoint.split(',') if step.strip()]
    print(f"Will save checkpoints at steps: {checkpoint_steps}")

    # For single task, we always split the data into train/validation
    print(f"Splitting single task dataset with train ratio: {cfg.train_ratio}")
    print(f"Training set will respect max_episodes_per_dataset={cfg.max_episodes_per_dataset}, max_size={cfg.max_size}")

    obs_type = utils.PRETRAINING_OBS_KEY_REGISTRY.get(cfg.obs_type, cfg.obs_type) # TODO unify with OBS_KEY_REGISTRY, generate dataset accordingly
    print(f"Observation key for the replay buffer:")
    utils.ColorPrint.blue(f"{obs_type}")
    
    utils.ColorPrint.green(f"Using observation key: {obs_type}")

    train_dataloader, valid_dataloader, _ = get_dataloaders(cfg, obs_type)

    # Initialize steps and epoch
    steps = 0
    start_epoch = 0
    
    # Initialize utilities for logging and checkpointing
    pretrained_path = None # TODO remove if not needed
    metrics_logger = MetricsLogger(
        use_wandb=cfg.use_wandb,
        log_every=getattr(cfg, 'log_every', None),
        log_frequency=cfg.log_frequency
    )
    model_checkpointer = ModelCheckpointer(
        save_path=cfg.save_path,
        dataset_config=cfg.dataset_config,
        cfg=cfg
    )

    
    # Load saved checkpoint if provided
    saved_args = None
    if cfg.resume_checkpoint:
        print(f"Loading checkpoint from {cfg.resume_checkpoint}")
        checkpoint = torch.load(cfg.resume_checkpoint, map_location=device)
        saved_args = checkpoint.get('args', {})
        print(f"Checkpoint args: {saved_args}")
    
        
        # Set steps and epoch from checkpoint if requested
        if cfg.continue_steps and 'steps' in checkpoint:
            steps = checkpoint['steps']
            print(f"Resuming from step: {steps}")
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
            print(f"Resuming from epoch: {start_epoch}")

    # Initialize wandb if enabled
    if cfg.use_wandb:
        print("Initializing Weights & Biases logging")
        wandb_config = OmegaConf.to_container(cfg, resolve=True)

        if cfg.resume_wandb_run:
            print(f"Resuming wandb run: {cfg.resume_wandb_run}")
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_run_name,
                id=cfg.resume_wandb_run,
                resume="must",
                config=wandb_config
            )
        else:
            print("Starting new wandb run")
            wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_run_name,
                config=wandb_config
            )

    # Get a batch from training data to initialize the agent
    batch = next(iter(train_dataloader))
    obs_shape = [*batch[0].shape[1:]]
    action_shape = [*batch[1].shape[1:]]

    # Log agent configuration
    utils.ColorPrint.blue(f"Initializing agent: {cfg.agent._target_}")
    utils.ColorPrint.blue(f"Observation shape: {obs_shape}")
    utils.ColorPrint.blue(f"Action shape: {action_shape}")

    if hasattr(cfg, 'wrapper') and cfg.wrapper is not None and cfg.wrapper._target_ != 'none':
        wrapper_cfg = cfg.wrapper
    else:
        wrapper_cfg = None

    # Initialize agent using Hydra
    taco_agent = make_wrapped_agent(obs_shape, action_shape, cfg.agent, wrapper_cfg)

    # Now that the agent is initialized with the loaded checkpoint, we're ready to continue training
    valid_iterator = iter(valid_dataloader)
    
    # Training loop with epochs
    for epoch in range(start_epoch, cfg.num_epochs):
        print(f"\n{'='*80}")
        print(f"Epoch: {epoch + 1}/{cfg.num_epochs}, Steps: {steps}")
        print(f"{'='*80}\n")
        
        for batch_idx, batch in enumerate(train_dataloader):
            
            # *** Validation evaluation step ***
            if (steps // cfg.batch_size) % cfg.eval_frequency == 0:
                taco_agent.train(False)  # Set to eval mode
                
                eval_metrics_sum = metrics_logger.create_eval_metrics_dict()
                num_eval_batches = 0
                
                # Evaluate on validation set
                for _ in range(cfg.eval_batches):
                    try:
                        eval_batch = next(valid_iterator)
                    except StopIteration:
                        valid_iterator = iter(valid_dataloader)
                        eval_batch = next(valid_iterator)
                    
                    obs, action, action_seq, reward, discount, next_obs, r_next_obs = utils.to_torch(
                        eval_batch, device)
                    
                    eval_metrics = taco_agent.evaluate_taco(obs, action, action_seq, r_next_obs, reward)
                    metrics_logger.accumulate_eval_metrics(eval_metrics_sum, eval_metrics, cfg)
                    num_eval_batches += 1
                
                # Average and display validation metrics
                eval_metrics_avg = metrics_logger.average_metrics(eval_metrics_sum, num_eval_batches)
                metrics_logger.print_eval_metrics(eval_metrics_avg, epoch + 1, steps)
                
                # Save best model if improved
                current_eval_loss = eval_metrics_avg['eval/total_loss']
                model_checkpointer.save_best_model(taco_agent, steps, epoch + 1, pretrained_path, current_eval_loss)
                
                taco_agent.train(True)  # Set back to train mode
            
            # *** Training step ***
            obs, action, action_seq, reward, discount, next_obs, r_next_obs = utils.to_torch(
                batch, device)
            metrics = taco_agent.update_taco(obs, action, action_seq, r_next_obs, reward)
            steps += cfg.batch_size
            
            # *** Logging ***
            metrics['steps'] = steps
            metrics['epoch'] = epoch + 1
            metrics['batch_idx'] = batch_idx
            
            # Add validation metrics if just computed
            if (steps // cfg.batch_size - 1) % cfg.eval_frequency == 0 and 'eval_metrics_avg' in locals():
                metrics.update(eval_metrics_avg)
            
            # Log to wandb
            metrics_logger.log_to_wandb(metrics)
            
            # Print training metrics
            metrics_logger.print_training_metrics(
                metrics, epoch + 1, batch_idx, steps, len(train_dataloader)
            )

            # *** Save model checkpoint ***
            if any(s <= steps < s + cfg.batch_size for s in checkpoint_steps):
                model_checkpointer.save_checkpoint(taco_agent, steps, epoch + 1, pretrained_path)
    
    # *** Save the final trained model ***
    model_checkpointer.save_final_model(taco_agent, steps, cfg.num_epochs, pretrained_path)
    
    print(f"Training completed after {cfg.num_epochs} epochs and {steps} steps")
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()