import torch
import torch.nn as nn
from agents.components import TACO
import torch.nn.functional as F
import utils
from .components import RandomShiftsAug, ProprioceptiveEncoder
import itertools
import utils
from typing import Tuple, Optional, Dict, Any, Union

class TACOWrapper:
    """Wraps any classical algorithm with TACO enhancements"""

    def __init__(self, base_algorithm, encoder_lr, feature_dim,
                 hidden_dim, reward, multistep, latent_a_dim, curl, 
                 pretrained_path=None, freeze_encoder=False, no_enc_auxiliary_losses=False, optimizer_type="adam"):
        
        self.base_algorithm = base_algorithm
        self.obs_type = self.base_algorithm.obs_type
        self.obs_shape = self.base_algorithm.obs_shape
        self.action_shape = self.base_algorithm.action_shape
        self.device = self.base_algorithm.device
        self.update_every_steps = self.base_algorithm.update_every_steps
        self.use_tb = self.base_algorithm.use_tb
        self.num_expl_steps = self.base_algorithm.num_expl_steps
        self.stddev_schedule = self.base_algorithm.stddev_schedule
        self.stddev_clip = self.base_algorithm.stddev_clip
        self.freeze_encoder = self.base_algorithm.freeze_encoder

        self.reward = reward
        self.curl = curl
        self.multistep = multistep
        self.optimizer_type = optimizer_type.lower()


        ### A heuristics to choose the dimensionality of latent actions
        if latent_a_dim == 'none':
            latent_a_dim = int(self.action_shape[0]*1.25)+1
        ### Create action embeddings
        self.act_tok = utils.ActionEncoding(self.action_shape[0], latent_a_dim, multistep)

        # Initialize encoder
        if hasattr(self.base_algorithm, 'encoder') and not isinstance(self.base_algorithm.encoder, nn.Identity): # Assume that every vision based algorithm has an encoder attribute, and prioprio based algorithm always use Identity
            self.encoder = self.base_algorithm.encoder
            self.load_encoder = False
        else:
            assert self.obs_type == 'proprio_obs', "With image observations, an encoder must be provided in the base algorithm."
            self.encoder = ProprioceptiveEncoder(self.obs_shape, feature_dim).to(self.device)
            self.base_algorithm.encoder = self.encoder  
            self.load_encoder = self.base_algorithm.load_encoder
            self.base_algorithm.build_actor_critic()  # We need to rebuild actor critic to match the new encoder output size

        
        self.actor = self.base_algorithm.actor
        self.critic = self.base_algorithm.critic
        if hasattr(self.base_algorithm, 'critic_target'):
            self.critic_target = self.base_algorithm.critic_target

        self.TACO = TACO(self.encoder.repr_dim, feature_dim, self.action_shape, latent_a_dim, hidden_dim, self.act_tok, self.encoder, self.multistep, self.device).to(self.device)
        self.freeze_encoder = freeze_encoder
        self.no_enc_auxiliary_losses = no_enc_auxiliary_losses
        ### State & Action Encoders
        parameters = itertools.chain(self.encoder.parameters(),
                                     self.act_tok.parameters(),
        )
        
        # Selezione dell'optimizer
        if self.optimizer_type == "adam":
            optimizer_class = torch.optim.Adam
        elif self.optimizer_type == "sgd":
            optimizer_class = torch.optim.SGD
        else:
            raise ValueError(f"Optimizer type '{self.optimizer_type}' not supported. Use 'adam' or 'sgd'.")
        
        if not self.freeze_encoder:
            self.encoder_opt = optimizer_class(parameters, lr=encoder_lr)
            if not self.no_enc_auxiliary_losses:
                self.taco_opt = optimizer_class(self.TACO.parameters(), lr=encoder_lr)

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        # data augmentation
        if hasattr(self.base_algorithm, 'aug'):
            self.aug = self.base_algorithm.aug
        elif self.obs_type == 'pixel_obs':
            self.aug = RandomShiftsAug(pad=4)
        else:
            self.aug = nn.Identity()

        if pretrained_path is not None and pretrained_path != 'none' and self.load_encoder:
            self._load_components(pretrained_path)
        if freeze_encoder:
            self._freeze_encoder()
        
        self.pretrained_path = pretrained_path
        self.train()
        utils.ColorPrint.green("Using TACO Wrapper")

    def _load_components(self, models_path: Union[str, Dict[str, Any]]):
        utils.ColorPrint.blue(f"Loading pretrained model from: {models_path}")

        if isinstance(models_path, str):
            checkpoint = torch.load(models_path, map_location=self.device, weights_only=False)
        else:
            checkpoint = models_path

        print("Loading encoder weights...")
        self.TACO.load_state_dict(checkpoint['taco'])
        utils.ColorPrint.green("✓  Loaded pretrained TACO model.")
        
        if isinstance(self.encoder, ProprioceptiveEncoder):
            self.encoder.load_state_dict(checkpoint['encoder'])
            utils.ColorPrint.green("✓  Loaded pretrained encoder.")
        
        self.act_tok.load_state_dict(checkpoint['act_tok'])
        utils.ColorPrint.green("✓  Loaded pretrained action tokenizer.")
            
    def _freeze_encoder(self):
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.act_tok.eval()
        for param in self.act_tok.parameters():
            param.requires_grad = False
        
        utils.ColorPrint.blue("🔒 Encoder components frozen")

    def train(self, training=True):
        self.training = training
        self.base_algorithm.train(training)
        self.encoder.train(training)
        self.TACO.train(training)
        self.act_tok.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]
    
    def update_taco(self, obs, action, action_seq, next_obs, reward):
        metrics = dict()
        
        obs_anchor = self.aug(obs.float())
        obs_pos = self.aug(obs.float())
        z_a = self.TACO.encode(obs_anchor)
        z_pos = self.TACO.encode(obs_pos, ema=True)
        ### Compute CURL loss
        if self.curl:
            logits = self.TACO.compute_logits(z_a, z_pos)
            labels = torch.arange(logits.shape[0]).long().to(self.device)
            curl_loss = self.cross_entropy_loss(logits, labels)
        else:
            curl_loss = torch.tensor(0.)
        
        ### Compute action encodings
        action_en = self.TACO.act_tok(action, seq=False) 
        action_seq_en = self.TACO.act_tok(action_seq, seq=True)
        
        ### Compute reward prediction loss
        if self.reward:
            reward_pred = self.TACO.reward(torch.concat([z_a, action_seq_en], dim=-1))
            reward_loss = F.mse_loss(reward_pred, reward)
        else:
            reward_loss = torch.tensor(0.)
        
        ### Compute TACO loss
        next_z = self.TACO.encode(self.aug(next_obs.float()), ema=True)
        curr_za = self.TACO.project_sa(z_a, action_seq_en) 
        logits = self.TACO.compute_logits(curr_za, next_z)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        taco_loss = self.cross_entropy_loss(logits, labels)
        
        if not self.freeze_encoder:
            self.taco_opt.zero_grad()
            (taco_loss + curl_loss + reward_loss).backward()
            self.taco_opt.step()
        if self.use_tb:
            metrics['reward_loss']  = reward_loss.item()
            metrics['curl_loss'] = curl_loss.item()
            metrics['taco_loss']  = taco_loss.item()
            metrics['total_loss'] = taco_loss.item() + curl_loss.item() + reward_loss.item()
        return metrics
    

    def update(self, replay_iter, step):
        metrics = dict()
        if step % self.update_every_steps != 0:
            return metrics

        metrics.update(self.base_algorithm.update(replay_iter, step))  # Update base algorithm components

        batch = next(replay_iter)
        obs, action, action_seq, reward, discount, next_obs, r_next_obs = utils.to_torch(
            batch, self.device)
        
        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()
        
        if self.no_enc_auxiliary_losses:
            return metrics
        
        metrics.update(self.update_taco(obs, action, action_seq, r_next_obs, reward))
        return metrics
    
    def evaluate_taco(self, obs, action, action_seq, next_obs, reward):
        with torch.no_grad():
            metrics = dict()
            metrics['batch_reward'] = reward.mean().item()
            
            obs_anchor = self.aug(obs.float())
            obs_pos = self.aug(obs.float())
            z_a = self.TACO.encode(obs_anchor)
            z_pos = self.TACO.encode(obs_pos, ema=True)
            ### Compute CURL loss
            if self.curl:
                logits = self.TACO.compute_logits(z_a, z_pos)
                labels = torch.arange(logits.shape[0]).long().to(self.device)
                curl_loss = self.cross_entropy_loss(logits, labels)
            else:
                curl_loss = torch.tensor(0.)
            
            ### Compute action encodings
            action_en = self.TACO.act_tok(action, seq=False) 
            action_seq_en = self.TACO.act_tok(action_seq, seq=True)
            
            ### Compute reward prediction loss
            if self.reward:
                reward_pred = self.TACO.reward(torch.concat([z_a, action_seq_en], dim=-1))
                reward_loss = F.mse_loss(reward_pred, reward)
            else:
                reward_loss = torch.tensor(0.)
            
            ### Compute TACO loss
            next_z = self.TACO.encode(self.aug(next_obs.float()), ema=True)
            curr_za = self.TACO.project_sa(z_a, action_seq_en) 
            logits = self.TACO.compute_logits(curr_za, next_z)
            labels = torch.arange(logits.shape[0]).long().to(self.device)
            taco_loss = self.cross_entropy_loss(logits, labels)
            
            if self.use_tb:
                metrics['reward_loss']  = reward_loss.item()
                metrics['curl_loss'] = curl_loss.item()
                metrics['taco_loss']  = taco_loss.item()
                metrics['total_loss'] = taco_loss.item() + curl_loss.item() + reward_loss.item()
        return metrics
