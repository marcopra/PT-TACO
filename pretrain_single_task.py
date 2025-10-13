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



def make_agent(obs_shape, action_shape, cfg):
    """Create agent based on configuration"""
    cfg.obs_shape = obs_shape
    cfg.action_shape = action_shape
    return hydra.utils.instantiate(cfg)


@hydra.main(config_path='cfgs', config_name='config_pretrain_ST')
def main(cfg: DictConfig):
    
    # Set device and seed
    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)
    
    # Compatibility checks
    assert cfg.multistep == cfg.nstep, f"Don't know the difference between nstep and multistep, set them to the same value"

    # Parse checkpoint steps from string to list of integers
    checkpoint_steps = [int(step) for step in cfg.checkpoint.split(',') if step.strip()]
    print(f"Will save checkpoints at steps: {checkpoint_steps}")

    # Format pretrained_path based on feature extractor
    pretrained_path = format_pretrained_path(cfg.feature_extractor)
    print(f"Using feature extractor: {cfg.feature_extractor}, pretrained_path: {pretrained_path}")

    # Handle dataset config for single task (no separate pretraining/test structure)
    config_path = Path(cfg.dataset_config)
    
    if config_path.suffix == '.json':
        # For JSON config, get dataset info from the config itself
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        # For single task, we expect the dataset path directly or in a simple format
        if 'dataset_path' in config_data:
            dataset_path = config_data['dataset_path']
            dataset_names = [extract_task_name_from_path(dataset_path)]
        elif 'pretraining_datasets' in config_data:
            dataset_dirs = config_data.get('pretraining_datasets', [])
            dataset_names = [extract_task_name_from_path(d) for d in dataset_dirs]
        else:
            dataset_names = ['single_task']
        print(f"Single task dataset config: {dataset_names}")
    else:
        # For folder-based config, assume the dataset is directly in the provided path
        dataset_path = cfg.dataset_config
        dataset_names = [extract_task_name_from_path(dataset_path)]
        print(f"Single task dataset path: {dataset_path}")
    
    # For single task, we always split the data into train/validation
    print(f"Splitting single task dataset with train ratio: {cfg.train_ratio}")
    print(f"Training set will respect max_episodes_per_dataset={cfg.max_episodes_per_dataset}, max_size={cfg.max_size}")

    if cfg.agent._target_ == "agents.taco_proprio_states.TACOAgent":
        observation_key = "proprio_observation"
    else:
        observation_key = "observation"
    utils.ColorPrint.green(f"Using observation key: {observation_key}")
    
    # Load training data with the specified constraints
    train_dataloader = load_single_task_dataset(
        config_or_path=cfg.dataset_config,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        max_episodes_per_dataset=cfg.max_episodes_per_dataset,
        max_size=cfg.max_size,
        homogeneous=cfg.homogeneous,
        train_ratio=cfg.train_ratio,
        use_training_split=True,  # Only get the training portion
        observation_key=observation_key
    )
    
    # Load validation data from the remaining data (no size constraints)
    valid_dataloader = load_single_task_dataset(
        config_or_path=cfg.dataset_config,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        max_episodes_per_dataset=None,  # Use all remaining episodes for validation
        max_size=None,  # Use all remaining data for validation
        homogeneous=cfg.homogeneous,
        train_ratio=cfg.train_ratio,
        use_training_split=False,  # Only get the validation portion
        observation_key=observation_key
    )
    
    # No separate test set for single task scenario
    test_dataloader = None

    # Initialize steps and epoch
    steps = 0
    epoch = 0
    
    # Initialize best model tracking
    best_eval_loss = float('inf')
    best_model_path = None
    
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
            epoch = checkpoint['epoch']
            print(f"Resuming from epoch: {epoch}")

    # Initialize wandb if enabled
    if cfg.use_wandb:
        print("Initializing Weights & Biases logging")
        wandb_config = OmegaConf.to_container(cfg, resolve=True)

        # Resume wandb run if ID is provided
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

    # Set up agent configuration dynamically based on feature extractor
    if cfg.feature_extractor != "conv":
        # For non-conv extractors, we need to use taco_resnet agent
        # Override the agent configuration
        cfg.agent._target_ = "agents.taco_resnet.TACOAgent"
        # Also override height and width in the config if they're being used
        cfg.height = cfg.get('height', 84)
        cfg.width = cfg.get('width', 84)
        
    # Initialize agent using Hydra
    taco_agent = make_agent(obs_shape, action_shape, cfg.agent)

    # Now that the agent is initialized with the loaded checkpoint, we're ready to continue training
    valid_iterator = iter(valid_dataloader)
    test_iterator = iter(test_dataloader) if test_dataloader is not None else None
    
    batch_count = 0
    while steps < cfg.total_steps:
        epoch += 1
        print(f"Epoch: {epoch}, Steps: {steps}/{cfg.total_steps}")
        
        for batch in train_dataloader:
            batch_count += 1
            
            # *** Validation evaluation step ***
            if (steps//cfg.batch_size) % cfg.eval_frequency == 0:
                taco_agent.train(False)  # Set to eval mode
                
                eval_metrics_sum = {
                    'eval/reward_loss': 0,
                    'eval/curl_loss': 0,
                    'eval/taco_loss': 0,
                    'eval/total_loss': 0,
                    'eval/batch_reward': 0,
                    'eval/avg_rew_pred_error_percentage': 0,
                    'eval/log_cosh': 0,
                    'eval/rel_error_filtered': 0,
                    'eval/smape': 0,
                }
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
                    
                    # Add eval/ prefix to metrics
                    eval_metrics_sum['eval/reward_loss'] += eval_metrics['reward_loss']
                    if 'curl_loss' in eval_metrics:
                        eval_metrics_sum['eval/curl_loss'] += eval_metrics['curl_loss']
                    eval_metrics_sum['eval/taco_loss'] += eval_metrics['taco_loss']
                    eval_metrics_sum['eval/batch_reward'] += eval_metrics['batch_reward']
                    if cfg.reward:
                        eval_metrics_sum['eval/avg_rew_pred_error_percentage'] += eval_metrics['avg_rew_pred_error_percentage']
                    eval_metrics_sum['eval/total_loss'] += eval_metrics['total_loss']
                    if 'log_cosh' in eval_metrics:
                        eval_metrics_sum['eval/log_cosh'] += eval_metrics['log_cosh']
                    if 'rel_error_filtered' in eval_metrics:
                        eval_metrics_sum['eval/rel_error_filtered'] += eval_metrics['rel_error_filtered']
                    if 'smape' in eval_metrics:
                        eval_metrics_sum['eval/smape'] += eval_metrics['smape']
                    num_eval_batches += 1
                
                # Average the validation metrics
                for key in eval_metrics_sum:
                    eval_metrics_sum[key] /= num_eval_batches
                
                # Display validation metrics table
                utils.print_metrics_table(eval_metrics_sum, f"Validation Metrics - Step {steps}")
                
                # Check if this is the best model based on validation loss
                current_eval_loss = eval_metrics_sum['eval/total_loss']
                if current_eval_loss < best_eval_loss:
                    print(f"New best model found! eval/total_loss: {current_eval_loss:.6f} (previous best: {best_eval_loss:.6f})")
                    best_eval_loss = current_eval_loss
                    
                    # Delete previous best model if it exists
                    if best_model_path is not None and os.path.exists(best_model_path):
                        print(f"Deleting previous best model: {best_model_path}")
                        os.remove(best_model_path)
                    
                    # Save new best model
                    curl_str = "curl" if cfg.curl else "nocurl"
                    reward_str = "rew" if cfg.reward else "norew"
                    optimizer_str = f"_{cfg.optimizer}" if cfg.optimizer != "adam" else ""
                    extractor_str = f"_{cfg.feature_extractor}" if cfg.feature_extractor != "conv" else ""
                    best_model_path = f"{cfg.save_path}/taco_ST{extractor_str}_{'_'.join(cfg.dataset_config.split('/')[1:])}_lr={cfg.lr}{optimizer_str}_ts={steps}_{curl_str}_{reward_str}_best.pt"
                    print(f"Saving new best model to {best_model_path} at step {steps}")
                    os.makedirs(cfg.save_path, exist_ok=True)
                    torch.save({
                        'encoder': taco_agent.encoder.state_dict(),
                        'taco': taco_agent.TACO.state_dict(),
                        'act_tok': taco_agent.act_tok.state_dict(),
                        'args': OmegaConf.to_container(cfg, resolve=True),
                        'steps': steps,
                        'epoch': epoch,
                        'best_eval_loss': best_eval_loss,
                        'feature_extractor': cfg.feature_extractor,
                        'pretrained_path': pretrained_path,
                    }, best_model_path)
                
                taco_agent.train(True)  # Set back to train mode
            
            # *** Test evaluation step (disabled for single task pretraining) ***
            # In single task pretraining, we don't have a separate test set
            # Validation data is used as the only evaluation source
            
            # *** Training step ***
            obs, action, action_seq, reward, discount, next_obs, r_next_obs = utils.to_torch(
                batch, device)
            # Training update
            metrics = taco_agent.update_taco(obs, action, action_seq, r_next_obs, reward)
            steps += cfg.batch_size
            
            # *** Logging ***
            # Log training metrics to wandb
            if cfg.use_wandb:
                metrics['steps'] = steps
                metrics['epoch'] = epoch
                # Log validation metrics if we just computed them
                if (steps//cfg.batch_size - 1) % cfg.eval_frequency == 0 and 'eval_metrics_sum' in locals():
                    metrics.update(eval_metrics_sum)
                wandb.log(metrics)
            
            # Print detailed metrics table every log_every batches
            if hasattr(cfg, 'log_every') and batch_count % cfg.log_every == 0:
                # Add step and epoch info to metrics for the table
                display_metrics = metrics.copy()
                display_metrics['steps'] = steps
                display_metrics['epoch'] = epoch
                display_metrics['batch_count'] = batch_count
                utils.print_metrics_table(display_metrics, f"Training Metrics - Step {steps}")
            
            # Print simple metrics every log_frequency batches (fallback)
            elif batch_count % cfg.log_frequency == 0:
                print(f"Steps: {steps}/{cfg.total_steps}, Batch: {batch_count}, Loss: {metrics.get('total_loss', 'N/A'):.6f}")

            # *** Save model checkpoint ***
            # Check if we need to save a checkpoint at this step
            if any(s <= steps < s + cfg.batch_size for s in checkpoint_steps):
                curl_str = "curl" if cfg.curl else "nocurl"
                reward_str = "rew" if cfg.reward else "norew"
                optimizer_str = f"_{cfg.optimizer}" if cfg.optimizer != "adam" else ""
                extractor_str = f"_{cfg.feature_extractor}" if cfg.feature_extractor != "conv" else ""
                checkpoint_path = f"{cfg.save_path}/taco_ST{extractor_str}_{'_'.join(cfg.dataset_config.split('/')[1:])}_lr={cfg.lr}{optimizer_str}_ts={steps}_{curl_str}_{reward_str}.pt"
                print(f"Saving checkpoint at step {steps} to {checkpoint_path}")
                os.makedirs(cfg.save_path, exist_ok=True)
                torch.save({
                    'encoder': taco_agent.encoder.state_dict(),
                    'taco': taco_agent.TACO.state_dict(),
                    'act_tok': taco_agent.act_tok.state_dict(),
                    'args': OmegaConf.to_container(cfg, resolve=True),  # Save configuration for easier loading
                    'steps': steps,
                    'epoch': epoch,
                    'feature_extractor': cfg.feature_extractor,
                    'pretrained_path': pretrained_path,
                }, checkpoint_path)
            
            if steps >= cfg.total_steps:
                break
    
    # *** Save the trained TACO agent ***
    print(f"Saving model to {cfg.save_path}")
    os.makedirs(cfg.save_path, exist_ok=True)
    curl_str = "curl" if cfg.curl else "nocurl"
    reward_str = "rew" if cfg.reward else "norew"
    optimizer_str = f"_{cfg.optimizer}" if cfg.optimizer != "adam" else ""
    extractor_str = f"_{cfg.feature_extractor}" if cfg.feature_extractor != "conv" else ""
    torch.save({
        'encoder': taco_agent.encoder.state_dict(),
        'taco': taco_agent.TACO.state_dict(),
        'act_tok': taco_agent.act_tok.state_dict(),
        'args': OmegaConf.to_container(cfg, resolve=True),  # Save configuration for easier loading
        'steps': steps,
        'epoch': epoch,
        'feature_extractor': cfg.feature_extractor,
        'pretrained_path': pretrained_path,
    }, f"{cfg.save_path}/taco_ST{extractor_str}_{'_'.join(cfg.dataset_config.split('/')[1:])}_lr={cfg.lr}{optimizer_str}_ts={cfg.total_steps}_{curl_str}_{reward_str}.pt")
    
    print(f"Training completed after {steps} steps and {epoch} epochs")
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()