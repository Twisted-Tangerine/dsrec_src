import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


# ============================================================================
# Feed-Forward Networks
# ============================================================================

class PointWiseFeedForward(nn.Module):
    """
    Point-wise feed-forward network with residual connection.
    
    Used in transformer-based sequential recommendation models.
    Reference: https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py
    """
    
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        """
        Args:
            inputs: Input tensor of shape (batch_size, seq_len, hidden_units)
        
        Returns:
            Output tensor with residual connection
        """
        # Conv1D requires (N, C, Length), so transpose
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # Transpose back to (N, Length, C)
        outputs += inputs  # Residual connection
        return outputs


# ============================================================================
# Loss Functions
# ============================================================================

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning representations.
    
    Computes bidirectional contrastive loss between two representations X and Y.
    """
    
    def __init__(self, tau=1, project=False, in_dim_1=None, in_dim_2=None, out_dim=None):
        """
        Args:
            tau: Temperature parameter for contrastive learning
            project: Whether to use projection layers
            in_dim_1: Input dimension for X
            in_dim_2: Input dimension for Y
            out_dim: Output dimension after projection
        """
        super().__init__()
        self.tau = tau
        self.project = project

        if project:
            if not in_dim_1:
                raise ValueError("in_dim_1 must be provided when project=True")
            self.x_projector = nn.Linear(in_dim_1, out_dim)
            self.y_projector = nn.Linear(in_dim_2, out_dim)

    def forward(self, X, Y):
        """
        Args:
            X: First representation tensor of shape (batch_size, hidden_size)
            Y: Second representation tensor of shape (batch_size, hidden_size)
        
        Returns:
            Contrastive loss value
        """
        if self.project:
            X = self.x_projector(X)
            Y = self.y_projector(Y)

        loss = self.compute_cl(X, Y) + self.compute_cl(Y, X)
        return loss

    def compute_cl(self, X, Y):
        """
        Compute contrastive loss between X and Y.
        
        Args:
            X: Tensor of shape (batch_size, hidden_size)
            Y: Tensor of shape (batch_size, hidden_size)
            tau: Temperature factor
        
        Returns:
            Contrastive loss tensor
        """
        # Compute cosine similarity matrix
        sim_matrix = F.cosine_similarity(X.unsqueeze(1), Y.unsqueeze(0), dim=2)
        pos = torch.exp(torch.diag(sim_matrix) / self.tau).unsqueeze(0)  # (1, batch_size)
        neg = torch.sum(torch.exp(sim_matrix / self.tau), dim=0) - pos    # (1, batch_size)
        loss = -torch.log(pos / neg)
        loss = loss.view(X.shape[0], -1)
        return loss


class ContrastiveLoss2(nn.Module):
    """
    Alternative contrastive loss implementation with cross-entropy.
    
    Uses a target distribution based on similarity between representations.
    """
    
    def __init__(self, tau=1):
        """
        Args:
            tau: Temperature parameter
        """
        super().__init__()
        self.temperature = tau

    def forward(self, X, Y):
        """
        Args:
            X: Tensor of shape (batch_size, dim)
            Y: Tensor of shape (batch_size, dim)
        
        Returns:
            Contrastive loss tensor
        """
        # Compute logits
        logits = (X @ Y.T) / self.temperature
        X_similarity = Y @ Y.T
        Y_similarity = X @ X.T

        # Target distribution based on similarity
        targets = F.softmax(
            (X_similarity + Y_similarity) / 2 * self.temperature, dim=-1
        )

        # Compute cross-entropy loss in both directions
        X_loss = self.cross_entropy(logits, targets, reduction='none')
        Y_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss = (Y_loss + X_loss) / 2.0

        return loss

    def cross_entropy(self, preds, targets, reduction='none'):
        """
        Compute cross-entropy loss.
        
        Args:
            preds: Predictions
            targets: Target distribution
            reduction: Reduction mode ('none' or 'mean')
        
        Returns:
            Loss tensor
        """
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()


def reg_params(model):
    """
    Compute L2 regularization loss for model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Total L2 norm squared of all parameters
    """
    reg_loss = 0
    for W in model.parameters():
        reg_loss += W.norm(2).square()
    return reg_loss


def cal_bpr_loss(anc_embeds, pos_embeds, neg_embeds):
    """
    Compute Bayesian Personalized Ranking (BPR) loss.
    
    Args:
        anc_embeds: Anchor embeddings
        pos_embeds: Positive embeddings
        neg_embeds: Negative embeddings
    
    Returns:
        BPR loss value
    """
    pos_preds = (anc_embeds * pos_embeds).sum(-1)
    neg_preds = (anc_embeds * neg_embeds).sum(-1)
    return torch.sum(F.softplus(neg_preds - pos_preds))


# ============================================================================
# Attention Mechanisms
# ============================================================================

class CalculateAttention(nn.Module):
    """
    Core attention computation module.
    
    Computes scaled dot-product attention with masking.
    """
    
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask):
        """
        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            mask: Attention mask (True for positions to mask)
        
        Returns:
            Attention output tensor
        """
        attention = torch.matmul(Q, torch.transpose(K, -1, -2))
        # Apply mask
        attention = attention.masked_fill_(mask, -1e9)
        # Scale and softmax
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention, V)
        return attention


class MultiCrossAttention(nn.Module):
    """
    Multi-head cross-attention module.
    
    In forward pass, the first argument is used to compute query,
    the second argument is used to compute key and value.
    """
    
    def __init__(self, hidden_size, all_head_size, head_num):
        """
        Args:
            hidden_size: Input dimension
            all_head_size: Output dimension
            head_num: Number of attention heads
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = all_head_size
        self.num_heads = head_num
        self.h_size = all_head_size // head_num

        assert all_head_size % head_num == 0, "all_head_size must be divisible by head_num"

        # Linear projections for Q, K, V
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # Normalization factor
        self.norm = sqrt(all_head_size)

    def forward(self, x, y, log_seqs):
        """
        Cross-attention: x is used for query, y is used for key and value.
        
        Args:
            x: Query input tensor of shape (batch_size, seq_len, hidden_size)
            y: Key/Value input tensor of shape (batch_size, seq_len, hidden_size)
            log_seqs: Sequence tensor for mask computation
        
        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size = x.size(0)

        # Project and reshape for multi-head attention
        # (B, S, D) -> (B, S, H, W) -> (B, H, S, W)
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # Create attention mask
        attention_mask = (log_seqs == 0).unsqueeze(1).repeat(1, log_seqs.size(1), 1).unsqueeze(1)

        # Compute attention
        attention = CalculateAttention()(q_s, k_s, v_s, attention_mask)
        
        # Reshape back: (B, H, S, W) -> (B, S, H*W)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        # Final output projection
        output = self.linear_output(attention)
        return output


class Attention(nn.Module):
    """
    Attention mechanism for sequence-to-sequence models.
    
    Supports dot-product and general attention methods.
    """
    
    def __init__(self, hidden_size, method="dot"):
        """
        Args:
            hidden_size: Hidden dimension size
            method: Attention method ("dot" or "general")
        """
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == "general":
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, query, key):
        """
        Args:
            query: Query tensor of shape (batch_size, hidden_size)
            key: Key tensor of shape (batch_size, seq_len, hidden_size)
        
        Returns:
            Attention weights of shape (batch_size, seq_len, 1)
        """
        if self.method == "dot":
            return self.dot_score(query, key)
        elif self.method == "general":
            return self.general_score(query, key)

    def dot_score(self, query, key):
        """Compute dot-product attention scores."""
        query = query.unsqueeze(2)  # (batch_size, hidden_size, 1)
        attn_energies = torch.bmm(key, query)  # (batch_size, seq_len, 1)
        attn_energies = attn_energies.squeeze(-1)  # (batch_size, seq_len)
        return F.softmax(attn_energies, dim=-1).unsqueeze(-1)  # (batch_size, seq_len, 1)

    def general_score(self, query, key):
        """Compute general attention scores."""
        query = self.Wa(query).unsqueeze(2)  # (batch_size, hidden_size, 1)
        attn_energies = torch.bmm(key, query).squeeze(-1)  # (batch_size, seq_len)
        return F.softmax(attn_energies, dim=-1).unsqueeze(-1)  # (batch_size, seq_len, 1)


# ============================================================================
# Graph Operations
# ============================================================================

class SpAdjEdgeDrop(nn.Module):
    """
    Sparse adjacency matrix edge dropout for graph neural networks.
    
    Randomly drops edges from a sparse adjacency matrix during training.
    """
    
    def __init__(self):
        super(SpAdjEdgeDrop, self).__init__()

    def forward(self, adj, keep_rate):
        """
        Args:
            adj: Sparse adjacency matrix
            keep_rate: Probability of keeping an edge (0.0 to 1.0)
        
        Returns:
            Sparse adjacency matrix with dropped edges
        """
        if keep_rate == 1.0:
            return adj
        
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        
        # Create random mask
        mask = (torch.rand(edgeNum) + keep_rate).floor().type(torch.bool)
        
        # Apply mask
        newVals = vals[mask]
        newIdxs = idxs[:, mask]
        
        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

# Maintain backward compatibility with old naming
Contrastive_Loss = ContrastiveLoss
Contrastive_Loss2 = ContrastiveLoss2
Multi_CrossAttention = MultiCrossAttention
