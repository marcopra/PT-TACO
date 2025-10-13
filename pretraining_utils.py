from replay_buffer import make_replay_loader
from pathlib import Path
import json
import random
import numpy as np
import tempfile
import os
from utils import *

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
                             homogeneous=False, train_ratio=0.8, use_training_split=True, observation_key='observation'):
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
            observation_key=observation_key
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
            observation_key=observation_key
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