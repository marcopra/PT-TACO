from collections import deque
from typing import Any, NamedTuple
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import mujoco
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)
from dm_env import StepType, specs
from PIL import Image
from envs.maze import MEDIUM_MAZE_RANDOM_INIT_FIXED_GOAL, MEDIUM_MAZE_FIXED_INIT_RANDOM_GOAL, MEDIUM_MAZE_FIXED_INIT_FIXED_GOAL

class ResizeRendering(gym.Wrapper):

    def __init__(self, env, resolution=224):
        super().__init__(env)
        self.resolution = resolution

    def render(self):
        img = super().render()

        # # Flip verticale per correggere l'orientamento (MuJoCo restituisce immagini capovolte)
        # img = np.flipud(img)

        # Convert numpy array to PIL Image
        img = Image.fromarray(img.astype(np.uint8))
        
        # Resize the image
        img_resized = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        
        # Convert back to numpy array
        return np.array(img_resized)
    
    def set_task(self, task):
        """Set the task for the environment."""
        # Set the task in the base environment
        self.env.set_task(task)
    
    def __getattr__(self, name):
        """Forward other attributes to the wrapped environment."""
        return getattr(self.env, name)

class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    pixel_obs: Any
    proprio_obs: Any
    action: Any
    success: Any = None

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, num_repeats, data_collection=False):
        super().__init__(env)
        self._num_repeats = num_repeats
        self.data_collection = data_collection
        self.obs_keys = None
        obs, info = self.env.reset()
        self.proprio_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._process_proprio_obs(obs).shape, dtype=np.float32)

    def _process_proprio_obs(self, obs):
        """Process proprioceptive observation, concatenating dict values if needed."""
    
        if isinstance(obs, dict):
            if self.obs_keys is None:
                self.obs_keys = []
                for key in obs.keys():  # Sort for consistent ordering
                    self.obs_keys.append(key)
                print(f"Proprio obs keys order: {self.obs_keys}") 

            # Concatenate all values in the dictionary
            arrays = []
            for key in self.obs_keys:
                arrays.append(obs[key].flatten())
            assert self.obs_keys == list(obs.keys()), f"Expected keys {self.obs_keys}, but got {list(obs.keys())}"  
            return np.concatenate(arrays, dtype=np.float32)
        else:
            return obs

    def step(self, action):
        reward = 0.0
        discount = 1.0
        done = False
        info = {}
        
        for i in range(self._num_repeats):
            obs, reward_step, terminated, truncated, info = self.env.step(action)
            # Handle success as a termination condition in MetaWorld
            done = terminated or truncated
            
            reward += reward_step * discount
            discount *= 0.99  # Standard discount factor
            
            if done:
                break
                
        # Convert gym step to dm_env format for compatibility
        if done:
            step_type = StepType.LAST
        else:
            step_type = StepType.MID
        image_obs = self.env.render()
        proprio_obs = self._process_proprio_obs(obs)
        return ExtendedTimeStep(
            step_type=step_type,
            reward=reward,
            discount=discount if not done else 0.0,
            pixel_obs=image_obs,  # Use image observations
            proprio_obs=proprio_obs,
            action=action,
            success=info['success'] if 'success' in info else terminated,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        image_obs = self.env.render()
        proprio_obs = self._process_proprio_obs(obs)
        # Convert gym reset to dm_env format
        return ExtendedTimeStep(
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            pixel_obs=image_obs,  # Use image observations
            proprio_obs=proprio_obs,
            action=np.zeros(self.env.action_space.shape, dtype=np.float32),
            success=False
        )
    
    @property
    def physics(self):
        """Forward physics attribute if available."""
        if hasattr(self.env, 'physics'):
            return self.env.physics
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute 'physics'")
    
    def __getattr__(self, name):
        """Forward other attributes to the wrapped environment."""
        return getattr(self.env, name)


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, num_frames):
        super().__init__(env)
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        
        # Update observation space to include stacked frames
        obs = env.reset()

        # Get the shape from the observation
        if isinstance(obs.pixel_obs, np.ndarray):
            self.orig_obs_shape = obs.pixel_obs.shape
        else:
            # Handle case where observation might be a different structure
            raise ValueError("Expected observation to be a numpy array")
        
        # Create a new stacked observation space
        channels = self.orig_obs_shape[2] * num_frames
        self.observation_space = gym.spaces.Box(
            low=0, 
            high=255, 
            shape=(channels, self.orig_obs_shape[0], self.orig_obs_shape[1]),
            dtype=np.uint8
        )


    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        # Stack frames along the channel dimension (axis 0 after transpose)
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(pixel_obs=obs)

    def _extract_pixels(self, obs):
        # Transform HWC to CHW format
        if isinstance(obs, np.ndarray):
            return obs.transpose(2, 0, 1).copy()
        else:
            raise ValueError("Expected observation to be a numpy array")

    def reset(self, **kwargs):
        time_step = self.env.reset(**kwargs)
        pixels = self._extract_pixels(time_step.pixel_obs)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self.env.step(action)
        pixels = self._extract_pixels(time_step.pixel_obs)
        self._frames.append(pixels)
        return self._transform_observation(time_step)
    
    @property
    def physics(self):
        """Forward physics attribute if available."""
        if hasattr(self.env, 'physics'):
            return self.env.physics
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute 'physics'")
    
    def __getattr__(self, name):
        """Forward other attributes to the wrapped environment."""
        return getattr(self.env, name)


class ActionDTypeWrapper(gym.Wrapper):
    def __init__(self, env, dtype=np.float32):
        super().__init__(env)
        original_space = env.action_space
        self.action_space = gym.spaces.Box(
            low=original_space.low.astype(dtype),
            high=original_space.high.astype(dtype),
            shape=original_space.shape,
            dtype=dtype
        )

    def step(self, action):
        action = action.astype(self.env.action_space.dtype)
        return self.env.step(action)
    
    def __getattr__(self, name):
        """Forward other attributes to the wrapped environment."""
        return getattr(self.env, name)


class PhysicsStateWrapper(gym.Wrapper):
    """Wrapper che simula l'interfacio physics per il relabelling come in CDMC."""
    
    def __init__(self, env):
        super().__init__(env)
        self._physics_state = None
    
    def _get_physics_state(self):
        """Estrae lo stato fisico dall'ambiente Gymnasium."""
        # Per PointMaze, usiamo la posizione e velocità come stato fisico
        if hasattr(self.env, 'unwrapped'):
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, 'point_env'):
                # PointMaze environment
                point_env = unwrapped.point_env
                qpos = point_env.data.qpos.copy()
                qvel = point_env.data.qvel.copy()
                return np.concatenate([qpos, qvel])
        
        # Fallback: usa l'osservazione propriocettiva se disponibile
        return self._physics_state if self._physics_state is not None else np.zeros(4)
    
    def _set_physics_state(self, state):
        """Imposta lo stato fisico nell'ambiente."""
        if hasattr(self.env, 'unwrapped'):
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, 'point_env'):
                # PointMaze environment
                point_env = unwrapped.point_env
                mid = len(state) // 2
                point_env.data.qpos[:] = state[:mid]
                point_env.data.qvel[:] = state[mid:]
                # Forward kinematics to update dependent variables
                mujoco.mj_forward(point_env.model, point_env.data)
    
    def reset(self,**kwargs):
        time_step = self.env.reset(**kwargs)
        self._physics_state = self._get_physics_state()
        return time_step
    
    def step(self, action):
        time_step = self.env.step(action)
        self._physics_state = self._get_physics_state()
        return time_step
    
    @property
    def physics(self):
        """Simula l'interfaccia physics di CDMC."""
        class PhysicsInterface:
            def __init__(self, wrapper):
                self.wrapper = wrapper
            
            def state(self):
                return self.wrapper._get_physics_state()
            
            def set_state(self, state):
                self.wrapper._set_physics_state(state)
            
            class ResetContext:
                def __init__(self, physics_interface):
                    self.physics = physics_interface
                    self.original_state = None
                
                def __enter__(self):
                    self.original_state = self.physics.state()
                    return self
                
                def __exit__(self, exc_type, exc_val, exc_tb):
                    if self.original_state is not None:
                        self.physics.set_state(self.original_state)
            
            def reset_context(self):
                return self.ResetContext(self)
        
        return PhysicsInterface(self)


class RewardSpecWrapper(gym.Wrapper):
    """Wrapper che aggiunge le specifiche per reward e discount compatibili con CDMC."""
    
    def __init__(self, env):
        super().__init__(env)
        # Verifica che sia un PointMaze environment
        if not hasattr(self.env, 'unwrapped') or not hasattr(self.env.unwrapped, 'compute_reward'):
            raise NotImplementedError("RewardSpecWrapper is currently only implemented for PointMaze environments")
    
    def reward_spec(self):
        """Specifica del reward per compatibilità con replay buffer CDMC."""
        return specs.Array(shape=(1,), dtype=np.float32, name='reward')
    
    def discount_spec(self):
        """Specifica del discount per compatibilità con replay buffer CDMC."""
        return specs.Array(shape=(1,), dtype=np.float32, name='discount')
    
    def compute_reward_from_state_and_action(self, physics_state, action, desired_goal=None):
        """Calcola il reward usando state e goal, senza fare uno step nell'environment."""
        unwrapped = self.env.unwrapped
        
        # Salva lo stato corrente e il goal corrente
        original_state = self.physics.state()
        original_goal = unwrapped.goal.copy()
        
        try:
            # Se desired_goal è fornito, usalo, altrimenti prendilo dagli ultimi elementi dello stato
            if desired_goal is not None:
                goal_to_use = desired_goal.copy()
            else:
                # Assumiamo che il goal sia negli ultimi 2 elementi del physics_state
                goal_to_use = physics_state[-2:].copy()
            
            # Estrai achieved_goal (posizione corrente) dai primi 2 elementi dello stato
            achieved_goal = physics_state[:2].copy()
            
            # Calcola il reward direttamente usando compute_reward
            if hasattr(unwrapped, 'compute_reward'):
                reward = unwrapped.compute_reward(achieved_goal, goal_to_use, {})
                return np.array([reward], dtype=np.float32)
            else:
                raise NotImplementedError("compute_reward method not found in environment")
            
        finally:
            # Ripristina lo stato originale (non necessario in questo caso ma per sicurezza)
            self.physics.set_state(original_state)
            # Ripristina sempre il goal originale
            unwrapped.goal = original_goal
            if hasattr(unwrapped, 'update_target_site_pos'):
                unwrapped.update_target_site_pos()
    
    def compute_reward_from_obs_dict(self, obs_dict, action=None):
        """Calcola il reward da un dizionario di osservazioni (formato PointMaze)."""
        if not all(key in obs_dict for key in ['achieved_goal', 'desired_goal']):
            raise ValueError("obs_dict must contain 'achieved_goal' and 'desired_goal' keys")
        
        achieved_goal = obs_dict['achieved_goal']
        desired_goal = obs_dict['desired_goal']
        
        # Per PointMaze, il reward dipende solo dalla posizione, non dallo stato fisico completo
        # quindi possiamo calcolare direttamente
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, 'compute_reward'):
            reward = unwrapped.compute_reward(achieved_goal, desired_goal, {})
            return np.array([reward], dtype=np.float32)
        else:
            raise NotImplementedError("compute_reward method not found in environment")
    
    
    def __getattr__(self, name):
        """Forward other attributes to the wrapped environment."""
        return getattr(self.env, name)


class ExtendedTimeStepWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        time_step = self.env.reset(**kwargs)
        return time_step

    def step(self, action):
        time_step = self.env.step(action)
        return time_step
    
    @property
    def physics(self):
        """Forward physics attribute if available."""
        if hasattr(self.env, 'physics'):
            return self.env.physics
        else:
            raise AttributeError(f"{self.__class__.__name__} has no attribute 'physics'")
    
    def __getattr__(self, name):
        """Forward other attributes to the wrapped environment."""
        return getattr(self.env, name)


def pixel_obs_spec(env):
    """Get pixel observation spec of the environment for agent initialization."""
    shape = env.observation_space.shape
    return specs.Array(shape, np.uint8, 'pixel_obs')

def proprio_obs_spec(env):
    """Get proprio observation spec of the environment for agent initialization."""
    shape = env.proprio_observation_space.shape
    return specs.Array(shape, np.float32, 'proprio_obs')


def action_spec(env):
    """Get action spec of the environment for agent initialization."""
    shape = env.action_space.shape
    min_action = env.action_space.low[0]
    max_action = env.action_space.high[0]
    return specs.BoundedArray(shape, np.float32, min_action, max_action, 'action')

def make(name, frame_stack=1, action_repeat=1, seed=None, resolution=224, random_init=True, randomize_goal=True, enable_relabelling=False):
    """
    Create a Gymnasium environment with wrappers.
    
    Args:
        name: Environment name (e.g., 'PointMaze_Medium-v3')
        frame_stack: Number of frames to stack
        action_repeat: Number of times to repeat each action
        seed: Random seed
        resolution: Image resolution
        random_init: Se True, usa posizioni iniziali casuali
        randomize_goal: Se True, usa goal casuali
        enable_relabelling: Se True, aggiunge i wrapper per il relabelling CDMC
    
    Returns:
        Wrapped environment
    """
    maze_map = None
    # PointMaze_MediumDense must be PointMaze_Medium_Diverse_GRDense
    if not random_init or not randomize_goal:   
        if 'PointMaze_Medium' in name:
            if random_init and not randomize_goal:
                name = name.replace('PointMaze_Medium', 'PointMaze_Medium_Diverse_GR')
                maze_map = MEDIUM_MAZE_RANDOM_INIT_FIXED_GOAL
            elif not random_init and randomize_goal:
                name = name.replace('PointMaze_Medium', 'PointMaze_Medium_Diverse_G')
                maze_map = MEDIUM_MAZE_FIXED_INIT_RANDOM_GOAL
            elif not random_init and not randomize_goal:
                name = name.replace('PointMaze_Medium', 'PointMaze_Medium_Diverse_GR')
                maze_map = MEDIUM_MAZE_FIXED_INIT_FIXED_GOAL
        else:
            raise ValueError("random_init and randomize_goal are only supported for 'PointMaze_Medium' environments")
        env = gym.make(name, render_mode='rgb_array', maze_map=maze_map)
 
    else:
        env = gym.make(name, render_mode='rgb_array')

    if seed is not None:
        env.reset(seed=seed)
    
    # Add wrappers
    env = ResizeRendering(env, resolution=resolution)   
    env = ActionDTypeWrapper(env, np.float32)
    
    # Add relabelling wrappers if requested
    if enable_relabelling:
        assert name.startswith('PointMaze'), "Relabelling wrappers are only implemented for PointMaze environments"
        env = PhysicsStateWrapper(env)
        env = RewardSpecWrapper(env)
    
    env = ActionRepeatWrapper(env, action_repeat)
    
    # Add frame stacking if requested
    env = FrameStackWrapper(env, frame_stack)
    
    env = ExtendedTimeStepWrapper(env)
    
    return env
