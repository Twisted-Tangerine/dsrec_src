"""
DSRec Model Implementation

This module contains:
1. Semantic Adapter classes (PLUS variants) that adapt semantic embeddings to backbone models
2. DSRec model classes that combine adapters with alignment loss for dual-space learning

The architecture follows: Backbone → BackbonePLUS → DSRec (with Alignment)
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
from src.models.backbone import SASRec, Bert4Rec, GRU4Rec
from src.models.components import ContrastiveLoss2


# ============================================================================
# Semantic Adapter Base Mixin
# ============================================================================

class SemanticAdapterMixin:
    """
    Base mixin class for semantic embedding adaptation.
    
    Provides common functionality for loading semantic embeddings and creating
    linear/nonlinear adapters to map them to the model's hidden space.
    """
    
    def _init_semantic_adapter(self, args):
        """
        Initialize semantic embedding adapter components.
        """
        self.hidden_size = args.hidden_size
        self.linear_dim = args.linear_dim
        self.nonlinear_dim = max(0, self.hidden_size - self.linear_dim)
        self.has_linear = self.linear_dim > 0
        self.has_nonlinear = self.nonlinear_dim > 0

        # Load semantic embeddings
        sem_item_emb = pickle.load(open(args.sem_emb_path, "rb"))
        # Add padding (index 0) and mask token (index -1)
        sem_item_emb = np.insert(sem_item_emb, 0, values=np.zeros((1, sem_item_emb.shape[1])), axis=0)
        sem_item_emb = np.insert(sem_item_emb, -1, values=np.zeros((1, sem_item_emb.shape[1])), axis=0)
        original_sem_dim = sem_item_emb.shape[1]
        
        self.item_emb = nn.Embedding.from_pretrained(torch.Tensor(sem_item_emb))
        if args.freeze_emb:
            self.item_emb.weight.requires_grad = False
        else:
            self.item_emb.weight.requires_grad = True

        # Initialize linear adapter
        if self.has_linear:
            self.linear_adapter = nn.Sequential(
                nn.Linear(original_sem_dim, self.linear_dim)
            )
            self.linear_norm = nn.LayerNorm(self.linear_dim)
        else:
            self.linear_adapter = None

        # Initialize nonlinear adapter
        if self.has_nonlinear:
            self.nonlinear_adapter = nn.Sequential(
                nn.Linear(original_sem_dim, max(256, original_sem_dim // 4)),
                nn.GELU(),
                nn.Linear(max(256, original_sem_dim // 4), self.nonlinear_dim)
            )
            self.nonlinear_norm = nn.LayerNorm(self.nonlinear_dim)
        else:
            self.nonlinear_adapter = None

        # Projection normalization and scaling
        self.proj_norm = nn.LayerNorm(self.hidden_size)
        self.proj_scale = nn.Parameter(torch.ones(self.hidden_size))

        # Track modules to skip during weight initialization
        if not hasattr(self, 'filter_init_modules'):
            self.filter_init_modules = []
        self.filter_init_modules.append("item_emb")

    def _get_embedding(self, log_seqs, apply_mask=True):
        """
        Get adapted embeddings from semantic embeddings.
        """
        item_seq_emb = self.item_emb(log_seqs)
        linear_emb = None
        nonlinear_emb = None

        # Apply linear adapter
        if self.has_linear:
            linear_emb = self.linear_adapter(item_seq_emb)
            linear_emb = self.linear_norm(linear_emb)

        # Apply nonlinear adapter
        if self.has_nonlinear:
            nonlinear_emb = self.nonlinear_adapter(item_seq_emb)
            nonlinear_emb = self.nonlinear_norm(nonlinear_emb)

        # Combine adapters
        if self.has_linear and self.has_nonlinear:
            item_seq_emb = torch.cat([linear_emb, nonlinear_emb], dim=-1)
        elif self.has_linear:
            item_seq_emb = linear_emb
        elif self.has_nonlinear:
            item_seq_emb = nonlinear_emb

        # Apply mask token replacement (for Bert4Rec)
        if apply_mask and hasattr(self, 'mask_embedding'):
            mask_token = getattr(self, "mask_token", None)
            if mask_token is not None and (log_seqs == mask_token).any():
                bool_mask = (log_seqs == mask_token)
                while bool_mask.dim() < item_seq_emb.dim():
                    bool_mask = bool_mask.unsqueeze(-1)
                mask_emb_expanded = self.mask_embedding.view(
                    *([1] * (item_seq_emb.dim() - 1)), -1
                ).expand_as(item_seq_emb)
                item_seq_emb = torch.where(bool_mask, mask_emb_expanded, item_seq_emb)

        # Apply projection normalization and scaling
        if self.has_linear and self.has_nonlinear:
            item_seq_emb = self.proj_norm(item_seq_emb) * self.proj_scale

        return item_seq_emb


# ============================================================================
# Semantic Adapter Classes
# ============================================================================

class SASRecPLUS(SASRec, SemanticAdapterMixin):
    """
    SASRec model with semantic embedding adapter.
    Extends SASRec backbone with semantic embeddings through linear/nonlinear adapters.
    """
    
    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self._init_semantic_adapter(args)
        self._init_weights()

    def _get_embedding(self, log_seqs, apply_mask=False):
        """Override to use semantic adapter embedding."""
        return SemanticAdapterMixin._get_embedding(self, log_seqs, apply_mask=apply_mask)

    def log2feats(self, log_seqs, positions):
        """Convert item sequences to feature representations."""
        seqs = self._get_embedding(log_seqs, apply_mask=False)
        seqs *= self.hidden_size ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)
        log_feats = self.backbone(seqs, log_seqs)
        return log_feats


class Bert4RecPLUS(Bert4Rec, SemanticAdapterMixin):
    """
    BERT4Rec model with semantic embedding adapter.
    Extends BERT4Rec backbone with semantic embeddings and mask token handling.
    """
    
    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self._init_semantic_adapter(args)
        
        # Initialize mask embedding for BERT-style masking
        self.mask_embedding = nn.Parameter(torch.randn(self.hidden_size) * 0.01)
        
        self._init_weights()

    def _get_embedding(self, log_seqs, apply_mask=True):
        """Override to use semantic adapter embedding with mask support."""
        return SemanticAdapterMixin._get_embedding(self, log_seqs, apply_mask=apply_mask)

    def log2feats(self, log_seqs, positions):
        """Convert item sequences to feature representations."""
        seqs = self._get_embedding(log_seqs, apply_mask=True)
        seqs *= self.hidden_size ** 0.5
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)
        log_feats = self.backbone(seqs, log_seqs)
        return log_feats


class GRU4RecPLUS(GRU4Rec, SemanticAdapterMixin):
    """
    GRU4Rec model with semantic embedding adapter.
    Extends GRU4Rec backbone with semantic embeddings (no positional encoding needed).
    """
    
    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self._init_semantic_adapter(args)
        self._init_weights()

    def _get_embedding(self, log_seqs, apply_mask=False):
        """Override to use semantic adapter embedding."""
        return SemanticAdapterMixin._get_embedding(self, log_seqs, apply_mask=apply_mask)

    def log2feats(self, log_seqs):
        """Convert item sequences to feature representations."""
        seqs = self._get_embedding(log_seqs, apply_mask=False)
        log_feats = self.backbone(seqs, log_seqs)
        return log_feats


# ============================================================================
# DSRec Alignment Mixin
# ============================================================================

class DSRecAlignmentMixin:
    """
    Mixin class for DSRec alignment loss computation.
    
    Handles:
    1. Loading pre-trained target (collaborative) embeddings
    2. Initializing alignment loss function
    3. Calculating frequency-weighted alignment loss during forward pass
    """
    
    def _init_alignment_module(self, args, item_weights=None):
        """
        Initialize alignment module for dual-space learning.
        """
        # Load pre-trained target embeddings (collaborative embeddings)
        target_path = "./data/{}/item_id_embeddings.pkl".format(args.dataset)
        target_item_emb = pickle.load(open(target_path, "rb"))
        # Add padding (index 0)
        target_item_emb = np.insert(target_item_emb, 0, values=np.zeros((1, target_item_emb.shape[1])), axis=0)
        
        self.target_emb = nn.Embedding.from_pretrained(torch.Tensor(target_item_emb))
        self.target_emb.weight.requires_grad = False  # Freeze target embeddings

        # Initialize alignment loss function
        self.align_loss_func = ContrastiveLoss2(args.tau)
        self.alpha = args.alpha

        # Register frequency-aware weights if provided
        if item_weights is not None:
            self.register_buffer('item_weights', item_weights)
            if hasattr(self, 'logger') and self.logger:
                self.logger.info("[Model] Frequency Weights successfully registered to Buffer.")
            else:
                print("[Model] Frequency Weights successfully registered to Buffer.")
        else:
            self.item_weights = None
            if hasattr(self, 'logger') and self.logger:
                self.logger.info("[Model] No Frequency Weights provided. Running in Uniform Mode.")
            else:
                print("[Model] No Frequency Weights provided. Running in Uniform Mode.")

        # Track modules to skip during weight initialization
        if not hasattr(self, 'filter_init_modules'):
            self.filter_init_modules = []
        self.filter_init_modules.append("target_emb")
        self._debug_printed = False

    def _calc_align_loss(self, pos, sem_embs):
        """
        Calculate alignment loss with optional frequency-aware weighting.
        """
        indices = (pos != 0)
        pos_ids = pos[indices]
        
        # Get target embeddings
        target_embs = self.target_emb(pos_ids)
        
        # Calculate per-sample loss vector
        raw_align_loss = self.align_loss_func(target_embs, sem_embs)
        
        # Apply frequency-aware weighting if available
        if self.item_weights is not None:
            batch_weights = self.item_weights[pos_ids]
            if not self._debug_printed:
                debug_msg = (
                    "\n" + ">"*20 + " RUNTIME DEBUG " + "<"*20 + "\n"
                    f"   [Debug] Batch Valid Items: {len(pos_ids)}\n"
                    f"   [Debug] Raw Loss Mean: {raw_align_loss.mean().item():.6f}\n"
                    f"   [Debug] Weight Stats: Min={batch_weights.min():.4f}, "
                    f"Max={batch_weights.max():.4f}, Mean={batch_weights.mean():.4f}\n"
                    f"   [Debug] Sample Weights: {batch_weights[:5].tolist()}\n"
                )
                weighted_mean = (raw_align_loss * batch_weights).mean().item()
                debug_msg += f"   [Debug] Weighted Loss Mean: {weighted_mean:.6f}\n" + ">"*55 + "\n"
                if hasattr(self, 'logger') and self.logger:
                    self.logger.info(debug_msg)
                else:
                    print(debug_msg)
                self._debug_printed = True
            # Element-wise multiply -> Mean
            align_loss = (raw_align_loss * batch_weights).mean()
        else:
            if not self._debug_printed:
                debug_msg = (f"\n[Debug] Running Uniform Alignment (No Weights). "
                            f"Raw Loss Mean: {raw_align_loss.mean().item():.6f}\n")
                if hasattr(self, 'logger') and self.logger:
                    self.logger.info(debug_msg)
                else:
                    print(debug_msg)
                self._debug_printed = True
            # Fallback (Standard DSRec)
            align_loss = raw_align_loss.mean()
            
        return align_loss


# ============================================================================
# DSRec Model Classes
# ============================================================================

class DSRecSASRec(SASRecPLUS, DSRecAlignmentMixin):
    
    def __init__(self, user_num, item_num, device, args, item_weights=None):
        super().__init__(user_num, item_num, device, args)
        if hasattr(args, 'logger'):
            self.logger = args.logger
        self._init_alignment_module(args, item_weights)
        self._init_weights()  # Re-run init to handle modules properly

    def forward(self, seq, pos, neg, positions, **kwargs):
        """Forward pass with recommendation loss and alignment loss."""
        # Main recommendation task
        loss = super().forward(seq, pos, neg, positions, **kwargs)

        # Alignment task
        indices = (pos != 0)
        sem_embs = self._get_embedding(pos[indices], apply_mask=False)
        align_loss = self._calc_align_loss(pos, sem_embs)
        loss += self.alpha * align_loss

        return loss


class DSRecBert4Rec(Bert4RecPLUS, DSRecAlignmentMixin):
    
    def __init__(self, user_num, item_num, device, args, item_weights=None):
        super().__init__(user_num, item_num, device, args)
        if hasattr(args, 'logger'):
            self.logger = args.logger
        self._init_alignment_module(args, item_weights)
        self._init_weights()

    def forward(self, seq, pos, neg, positions, **kwargs):
        loss = super().forward(seq, pos, neg, positions, **kwargs)

        indices = (pos != 0)
        sem_embs = self._get_embedding(pos[indices], apply_mask=True)
        align_loss = self._calc_align_loss(pos, sem_embs)
        loss += self.alpha * align_loss

        return loss


class DSRecGRU4Rec(GRU4RecPLUS, DSRecAlignmentMixin):
    
    def __init__(self, user_num, item_num, device, args, item_weights=None):
        super().__init__(user_num, item_num, device, args)
        if hasattr(args, 'logger'):
            self.logger = args.logger
        self._init_alignment_module(args, item_weights)
        self._init_weights()

    def forward(self, seq, pos, neg, positions, **kwargs):
        loss = super().forward(seq, pos, neg, positions, **kwargs)

        indices = (pos != 0)
        sem_embs = self._get_embedding(pos[indices], apply_mask=False)
        align_loss = self._calc_align_loss(pos, sem_embs)
        loss += self.alpha * align_loss

        return loss
