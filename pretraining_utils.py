from replay_buffer import make_replay_loader
from pathlib import Path
import json
import random
import numpy as np
import tempfile
import os
from utils import *
import wandb

def format_pretrained_path(feature_extractor):
    """Format pretrained_path based on feature extractor type"""
    if feature_extractor == "conv":
        return None
    
    # Map feature extractor names to formatted paths
    extractor_map = {
        "vit_s": "vit_s_scratch",
        "vit_b": "vit_b_scratch",
        "vit_l": "vit_l_scratch", 
        "resnet18": "resnet18_l5_scratch",
        "resnet18p": "resnet18_l5_pretrained",
        "resnet50": "resnet50_l5_scratch",
        "resnet50p": "resnet50_l5_pretrained",
        "r3m": "r3m",
        "mvp": "mvp"
    }
    
    return extractor_map.get(feature_extractor, f"{feature_extractor}_scratch")


def extract_dataset_name_from_path(dataset_config):
    config_path = Path(dataset_config)

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
        dataset_path = dataset_config
        dataset_names = [extract_task_name_from_path(dataset_path)]
        print(f"Single task dataset path: {dataset_path}")

def extract_task_name_from_path(dataset_path):
    """Extract task name from dataset path (same logic as generate_config.py)"""
    dataset_path = Path(dataset_path)
    folder_name = dataset_path.name
    
    # Extract task name from folder name 
    parts = folder_name.split('_')
    if parts:
        # Take the first part which should contain the task name
        task_part = parts[0]
        # Remove version suffixes like -v2, -v3
        task_name = task_part.replace('-v2', '').replace('-v3', '')
        return task_name
    
    # Fallback to using the full folder name if parsing fails
    return folder_name

def count_transitions_in_episodes(episode_files):
    """Count total transitions in a list of episode files"""
    total_transitions = 0
    for episode_file in episode_files:
        try:
            episode_file = Path(episode_file)
            with episode_file.open('rb') as f:
                episode = np.load(f)
                obs_key = next(iter(episode.keys()))
                episode_len = episode[obs_key].shape[0] - 1
                total_transitions += episode_len
        except Exception as e:
            print(f"Warning: Could not count transitions in {episode_file}: {e}")
            continue
    return total_transitions

def select_episodes_homogeneous(dataset_paths, max_episodes_per_dataset, max_size=None, is_test=False):
    """Select episodes from datasets to ensure homogeneous distribution"""
    all_episodes = []
    dataset_info = []
    
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        episode_files = list(dataset_path.glob('*.npz'))
        
        if max_episodes_per_dataset is not None:
            episode_files = episode_files[:max_episodes_per_dataset]
        
        transitions_count = count_transitions_in_episodes(episode_files)
        dataset_info.append({
            'path': dataset_path,
            'episodes': episode_files,
            'transitions': transitions_count
        })
        all_episodes.extend(episode_files)
    
    total_transitions = sum(info['transitions'] for info in dataset_info)
    if max_size is None:
        max_size = total_transitions
    if not is_test:
        if total_transitions > max_size:       
            ColorPrint.yellow(f"Warning: Total transitions ({total_transitions}) exceed max_size ({max_size})")
        transitions_per_dataset = max_size // len(dataset_paths)
        ColorPrint.green(f"Homogeneous loading: {transitions_per_dataset} transitions per dataset")
        
        selected_episodes = []
        actual_transitions_loaded = 0
        
        for info in dataset_info:
            dataset_episodes = info['episodes'].copy()
            random.shuffle(dataset_episodes)
            
            current_transitions = 0
            dataset_selected = []
            
            for episode_file in dataset_episodes:
                episode_transitions = count_transitions_in_episodes([episode_file])
                
                if current_transitions + episode_transitions <= transitions_per_dataset:
                    dataset_selected.append(episode_file)
                    current_transitions += episode_transitions
                elif len(dataset_selected) == 0:
                    dataset_selected.append(episode_file)
                    current_transitions += episode_transitions
                    ColorPrint.yellow(f"Added episode with {episode_transitions} transitions (exceeds per-dataset limit of {transitions_per_dataset})")
                    break
                else:
                    break
            
            selected_episodes.extend(dataset_selected)
            actual_transitions_loaded += current_transitions
            print(f"Dataset {info['path'].name}: selected {len(dataset_selected)} episodes with {current_transitions} transitions")
        
        print(f"Total selected: {len(selected_episodes)} episodes with {actual_transitions_loaded} transitions")
        return selected_episodes
    else:
        if not is_test:
            ColorPrint.green(f"Using all {total_transitions} transitions from {len(dataset_paths)} datasets")
        return all_episodes

def create_symlink_with_proper_naming(episode_file, dest_dir, counter):
    """Create symlink to episode file with proper naming format for replay_buffer"""
    episode_file = Path(episode_file)
    
    # Count transitions in the episode to get episode length
    try:
        with episode_file.open('rb') as f:
            episode = np.load(f)
            obs_key = next(iter(episode.keys()))
            episode_len = episode[obs_key].shape[0] - 1
    except Exception as e:
        print(f"Warning: Could not process episode {episode_file}: {e}")
        return None
    
    # Create proper filename format that replay_buffer expects: {timestamp}_{eps_idx}_{eps_len}.npz
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    proper_name = f"{timestamp}_{counter:06d}_{episode_len}.npz"
    dest_path = dest_dir / proper_name
    
    # Create symlink instead of copying
    os.symlink(episode_file.absolute(), dest_path)
    return dest_path

def can_use_direct_loading(dataset_paths, selected_episodes, max_episodes_per_dataset, homogeneous):
    """Check if we can use direct loading without temp directory"""
    # If we're using all episodes from all datasets without filtering, we can load directly
    if not homogeneous and max_episodes_per_dataset is None:
        return True, dataset_paths[0] if len(dataset_paths) == 1 else None
    
    # If all selected episodes are from a single dataset and we're not filtering
    if len(dataset_paths) == 1 and not homogeneous:
        dataset_path = Path(dataset_paths[0])
        all_episodes_in_dataset = list(dataset_path.glob('*.npz'))
        if len(selected_episodes) == len(all_episodes_in_dataset):
            return True, dataset_path
    
    return False, None

def load_unified_dataset(config_or_path, batch_size=32, num_workers=4,
                         nstep=3, multistep=3, discount=0.99, 
                         max_episodes_per_dataset=8, max_size=None,
                         homogeneous=False, is_test=False):
    """
    Load episodes from datasets specified in config or folder structure
    
    Args:
        config_or_path: Either path to config.json or folder with pretraining_datasets/test_dataset
        max_episodes_per_dataset: Maximum episodes to load per dataset
        max_size: Maximum size of replay buffer (only for training)
        homogeneous: Whether to load transitions evenly across datasets
        is_test: Whether this is test dataset (affects max_size behavior)
    """
    config_path = Path(config_or_path)
    
    # Determine if it's a config file or folder
    if config_path.suffix == '.json':
        ColorPrint.blue(f"Loading from config file: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        if is_test:
            dataset_paths = config.get('test_datasets', [])
        else:
            dataset_paths = config.get('pretraining_datasets', [])
    else:
        ColorPrint.blue(f"Loading from folder structure: {config_path}")
        if is_test:
            datasets_dir = config_path / "test_dataset"
        else:
            datasets_dir = config_path / "pretraining_datasets"
        
        dataset_paths = [str(d) for d in datasets_dir.iterdir() if d.is_dir()]
    
    print(f"Found {len(dataset_paths)} dataset directories")
    
    if homogeneous:
        ColorPrint.green("Using homogeneous dataset loading")
        selected_episodes = select_episodes_homogeneous(
            dataset_paths, max_episodes_per_dataset, max_size, is_test
        )
    else:
        selected_episodes = []
        for dataset_path in dataset_paths:
            dataset_path = Path(dataset_path)
            episode_files = list(dataset_path.glob('*.npz'))
            
            if max_episodes_per_dataset is not None:
                episode_files = episode_files[:max_episodes_per_dataset]
            
            selected_episodes.extend(episode_files)
    
    print(f"Selected {len(selected_episodes)} episodes total")
    
    # Check if we can use direct loading
    can_direct_load, direct_path = can_use_direct_loading(
        dataset_paths, selected_episodes, max_episodes_per_dataset, homogeneous
    )
    
    if can_direct_load and direct_path:
        ColorPrint.green(f"Using direct loading from {direct_path}")
        # Use direct loading like the old implementation
        loader = make_replay_loader(
            replay_dir=direct_path,
            max_size=1000000 if max_size is None else max_size,
            batch_size=batch_size,
            num_workers=num_workers,
            save_snapshot=True,
            nstep=nstep,
            multistep=multistep,
            discount=discount
        )
        return loader
    
    # If we can't use direct loading, create temporary directory with symlinks
    ColorPrint.yellow("Creating temporary directory with symlinks (no disk space used)")
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Temporary directory created at: {temp_dir}")
    
    try:
        # Create symlinks to selected episodes with proper naming
        counter = 0
        for episode_file in selected_episodes:
            symlink_path = create_symlink_with_proper_naming(episode_file, temp_dir, counter)
            if symlink_path is not None:
                counter += 1
        
        # Create loader
        loader = make_replay_loader(
            replay_dir=temp_dir,
            max_size=1000000 if max_size is None else max_size,
            batch_size=batch_size,
            num_workers=num_workers,
            save_snapshot=True,
            nstep=nstep,
            multistep=multistep,
            discount=discount
        )
        
        return loader
    finally:
        # Cleanup temporary directory and symlinks
        import shutil
        try:
            shutil.rmtree(temp_dir)
            print(f"Temporary directory {temp_dir} cleaned up successfully")
        except Exception as e:
            ColorPrint.yellow(f"Warning: Could not cleanup temporary directory {temp_dir}: {e}")

def load_single_task_dataset(config_or_path, batch_size=32, num_workers=4,
                             nstep=3, multistep=3, discount=0.99, 
                             max_episodes_per_dataset=8, max_size=None,
                             homogeneous=False, train_ratio=0.8, use_training_split=True, obs_type='observation'):
    """
    Load single task dataset and split into train/validation
    
    Args:
        config_or_path: Either path to config.json or folder with dataset
        max_episodes_per_dataset: Maximum episodes to load per dataset
        max_size: Maximum size of replay buffer (only applied to training split)
        homogeneous: Whether to load transitions evenly across datasets
        train_ratio: Ratio of data to use for training (rest for validation)
        use_training_split: If True, return training split; if False, return validation split
    """
    config_path = Path(config_or_path)
    
    # Determine if it's a config file or folder
    if config_path.suffix == '.json':
        ColorPrint.blue(f"Loading single task from config file: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # For single task, check different possible config formats
        if 'dataset_path' in config:
            dataset_paths = [config['dataset_path']]
        elif 'pretraining_datasets' in config:
            dataset_paths = config.get('pretraining_datasets', [])
        else:
            raise ValueError("Config must contain 'dataset_path' or 'pretraining_datasets'")
    else:
        ColorPrint.blue(f"Loading single task from folder: {config_path}")
        dataset_paths = [str(config_path)]
    
    print(f"Single task dataset paths: {dataset_paths}")
    
    # Collect all episodes from all datasets
    all_episodes = []
    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        episode_files = list(dataset_path.glob('*.npz'))
        
        if max_episodes_per_dataset is not None:
            episode_files = episode_files[:max_episodes_per_dataset]
        
        all_episodes.extend(episode_files)
    
    print(f"Found {len(all_episodes)} total episodes")
    
    # Shuffle episodes to ensure random distribution
    random.shuffle(all_episodes)
    
    # Split episodes based on train_ratio
    split_idx = int(len(all_episodes) * train_ratio)
    
    if use_training_split:
        selected_episodes = all_episodes[:split_idx]
        split_type = "training"
        effective_max_size = max_size  # Apply max_size only to training
    else:
        selected_episodes = all_episodes[split_idx:]
        split_type = "validation"
        effective_max_size = None  # No size limit for validation
    
    print(f"Selected {len(selected_episodes)} episodes for {split_type} split")
    
    if len(selected_episodes) == 0:
        raise ValueError(f"No episodes selected for {split_type} split. Check train_ratio and dataset size.")
    
    # For single dataset with direct loading, check if we can avoid temp directory
    if len(dataset_paths) == 1 and len(selected_episodes) == len(all_episodes):
        # Using all episodes from single dataset - direct loading possible
        dataset_path = Path(dataset_paths[0])
        ColorPrint.green(f"Using direct loading from {dataset_path} for {split_type}")
        
        loader = make_replay_loader(
            replay_dir=dataset_path,
            max_size=1000000 if effective_max_size is None else effective_max_size,
            batch_size=batch_size,
            num_workers=num_workers,
            save_snapshot=True,
            nstep=nstep,
            multistep=multistep,
            discount=discount,
            obs_type=obs_type
        )
        return loader
    
    # Need to create temporary directory with symlinks for split
    ColorPrint.yellow(f"Creating temporary directory with symlinks for {split_type} split")
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Temporary directory created at: {temp_dir}")
    
    try:
        # Create symlinks to selected episodes with proper naming
        counter = 0
        for episode_file in selected_episodes:
            symlink_path = create_symlink_with_proper_naming(episode_file, temp_dir, counter)
            if symlink_path is not None:
                counter += 1
        
        # Create loader
        loader = make_replay_loader(
            replay_dir=temp_dir,
            max_size=1000000 if effective_max_size is None else effective_max_size,
            batch_size=batch_size,
            num_workers=num_workers,
            save_snapshot=True,
            nstep=nstep,
            multistep=multistep,
            discount=discount,
            obs_type=obs_type
        )
        
        return loader
    finally:
        # Cleanup temporary directory and symlinks
        import shutil
        try:
            shutil.rmtree(temp_dir)
            print(f"Temporary directory {temp_dir} cleaned up successfully")
        except Exception as e:
            ColorPrint.yellow(f"Warning: Could not cleanup temporary directory {temp_dir}: {e}")

def get_dataloaders(cfg, obs_type, multi_task=False):
    """Get training and validation dataloaders for single task pretraining"""
    train_loader = load_single_task_dataset(
        config_or_path=cfg.dataset_config,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        nstep=cfg.nstep,
        multistep=cfg.multistep,
        discount=cfg.discount,
        max_episodes_per_dataset=cfg.max_episodes_per_dataset,
        max_size=cfg.max_size,
        homogeneous=cfg.homogeneous,
        train_ratio=cfg.train_ratio,
        use_training_split=True,
        obs_type=obs_type
    )
    
    valid_loader = load_single_task_dataset(
        config_or_path=cfg.dataset_config,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        nstep=cfg.nstep,
        multistep=cfg.multistep,
        discount=cfg.discount,
        max_episodes_per_dataset=cfg.max_episodes_per_dataset,
        max_size=None,  # No size limit for validation
        homogeneous=cfg.homogeneous,
        train_ratio=cfg.train_ratio,
        use_training_split=False,
        obs_type=obs_type
    )
    
    if multi_task:
        raise NotImplementedError("Multi-task loading not implemented in this function")
    return train_loader, valid_loader, None  # No test loader for single task

class MetricsLogger:
    """Gestisce la raccolta, aggregazione e logging delle metriche"""
    
    def __init__(self, use_wandb=False, log_every=None, log_frequency=100):
        self.use_wandb = use_wandb
        self.log_every = log_every
        self.log_frequency = log_frequency
        self.eval_metrics_keys = [
            'eval/reward_loss',
            'eval/curl_loss', 
            'eval/taco_loss',
            'eval/total_loss',
            'eval/batch_reward',
        ]
    
    def create_eval_metrics_dict(self):
        """Crea dizionario per accumulare metriche di valutazione"""
        return {key: 0 for key in self.eval_metrics_keys}
    
    def accumulate_eval_metrics(self, metrics_sum, eval_metrics, cfg):
        """Accumula metriche di valutazione da un batch"""
        metrics_sum['eval/reward_loss'] += eval_metrics['reward_loss']
        if 'curl_loss' in eval_metrics:
            metrics_sum['eval/curl_loss'] += eval_metrics['curl_loss']
        metrics_sum['eval/taco_loss'] += eval_metrics['taco_loss']
        metrics_sum['eval/batch_reward'] += eval_metrics['batch_reward']
        metrics_sum['eval/total_loss'] += eval_metrics['total_loss']
    
    
    def average_metrics(self, metrics_sum, num_batches):
        """Calcola la media delle metriche"""
        return {key: value / num_batches for key, value in metrics_sum.items()}
    
    def log_to_wandb(self, metrics):
        """Log metriche su wandb"""
        if self.use_wandb:
            wandb.log(metrics)
    
    def print_training_metrics(self, metrics, epoch, batch_idx, steps, total_batches=None):
        """Stampa metriche di training"""
        if self.log_every and (batch_idx + 1) % self.log_every == 0:
            display_metrics = metrics.copy()
            display_metrics['steps'] = steps
            display_metrics['epoch'] = epoch
            display_metrics['batch_idx'] = batch_idx + 1
            print_metrics_table(display_metrics, f"Training Metrics - Epoch {epoch}, Batch {batch_idx + 1}")
        elif (batch_idx + 1) % self.log_frequency == 0:
            batch_info = f"/{total_batches}" if total_batches else ""
            print(f"Epoch: {epoch}/{epoch}, Batch: {batch_idx + 1}{batch_info}, Steps: {steps}, Loss: {metrics.get('total_loss', 'N/A'):.6f}")
    
    def print_eval_metrics(self, metrics, epoch, steps):
        """Stampa metriche di valutazione"""
        
        print_metrics_table(metrics, f"Validation Metrics - Epoch {epoch}, Step {steps}")


class ModelCheckpointer:
    """Gestisce il salvataggio dei checkpoint e del miglior modello"""
    
    def __init__(self, save_path, dataset_config, cfg):
        self.save_path = save_path
        self.dataset_config = dataset_config
        self.cfg = cfg
        self.best_eval_loss = float('inf')
        self.best_model_path = None
        self.dataset_name= '_'.join(self.dataset_config.split('/')[self.dataset_config.split('/').index('dataset') + 1:])
        self.curl_str = "curl" if self.cfg.curl else "nocurl"
        self.reward_str = "rew" if self.cfg.reward else "norew"
        self.optimizer_str = f"_{self.cfg.optimizer}" if self.cfg.optimizer != "adam" else ""
        # extractor_str = f"_{self.cfg.feature_extractor}" if self.cfg.feature_extractor != "conv" else ""
        # Crea directory di salvataggio
        os.makedirs(save_path, exist_ok=True)
    
    def _create_checkpoint_dict(self, agent, steps, epoch, pretrained_path, best_eval_loss=None):
        """Crea dizionario con i dati del checkpoint"""
        checkpoint = {
            'encoder': agent.encoder.state_dict(),
            'taco': agent.TACO.state_dict(),
            'act_tok': agent.act_tok.state_dict(),
            'args': OmegaConf.to_container(self.cfg, resolve=True),
            'steps': steps,
            'epoch': epoch,
            # 'feature_extractor': self.cfg.feature_extractor,
            'pretrained_path': pretrained_path,
        }
        
        if best_eval_loss is not None:
            checkpoint['best_eval_loss'] = best_eval_loss
        
        return checkpoint
    
    def save_checkpoint(self, agent, steps, epoch, pretrained_path):
        """Salva checkpoint regolare"""


        checkpoint_path = f"{self.save_path}/taco_ST_{self.dataset_name}_lr{self.cfg.lr}{self.optimizer_str}_ts{steps}_{self.curl_str}_{self.reward_str}.pt"

        print(f"Saving checkpoint at step {steps} to {checkpoint_path}")
        
        checkpoint = self._create_checkpoint_dict(agent, steps, epoch, pretrained_path)
        torch.save(checkpoint, checkpoint_path)
        
        return checkpoint_path
    
    def save_best_model(self, agent, steps, epoch, pretrained_path, eval_loss):
        """Salva il miglior modello se migliora la loss di validazione"""
        if eval_loss >= self.best_eval_loss:
            return False
        
        print(f"New best model found! eval/total_loss: {eval_loss:.6f} (previous best: {self.best_eval_loss:.6f})")
        self.best_eval_loss = eval_loss
        
        # Elimina modello precedente
        if self.best_model_path is not None and os.path.exists(self.best_model_path):
            print(f"Deleting previous best model: {self.best_model_path}")
            os.remove(self.best_model_path)


        self.best_model_path = f"{self.save_path}/taco_ST_{self.dataset_name}_lr{self.cfg.lr}{self.optimizer_str}_ts{steps}_{self.curl_str}_{self.reward_str}_best.pt"

        print(f"Saving new best model to {self.best_model_path} at step {steps}")
        
        checkpoint = self._create_checkpoint_dict(agent, steps, epoch, pretrained_path, self.best_eval_loss)
        torch.save(checkpoint, self.best_model_path)
        
        return True
    
    def save_final_model(self, agent, steps, epoch, pretrained_path):
        """Salva il modello finale a fine training"""
       
        final_path = f"{self.save_path}/taco_ST_{self.dataset_name}_lr{self.cfg.lr}{self.optimizer_str}_ts{steps}_{self.curl_str}_{self.reward_str}_final.pt"

        print(f"Saving final model to {final_path}")
        
        checkpoint = self._create_checkpoint_dict(agent, steps, epoch, pretrained_path)
        torch.save(checkpoint, final_path)
        
        return final_path