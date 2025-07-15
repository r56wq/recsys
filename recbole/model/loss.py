# @Time   : 2020/6/26
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE:
# @Time   : 2020/8/7, 2021/12/22
# @Author : Shanlei Mu, Gaowei Zhang
# @Email  : slmu@ruc.edu.cn, 1462034631@qq.com


"""
recbole.model.loss
#######################
Common Loss in recommender system
"""

import torch
import torch.nn as nn


class BPRLoss(nn.Module):
    """BPRLoss, based on Bayesian Personalized Ranking

    Args:
        - gamma(float): Small value to avoid division by zero

    Shape:
        - Pos_score: (N)
        - Neg_score: (N), same shape as the Pos_score
        - Output: scalar.

    Examples::

        >>> loss = BPRLoss()
        >>> pos_score = torch.randn(3, requires_grad=True)
        >>> neg_score = torch.randn(3, requires_grad=True)
        >>> output = loss(pos_score, neg_score)
        >>> output.backward()
    """

    def __init__(self, gamma=1e-10):
        super(BPRLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pos_score, neg_score):
        loss = -torch.log(self.gamma + torch.sigmoid(pos_score - neg_score)).mean()
        return loss


class RegLoss(nn.Module):
    """RegLoss, L2 regularization on model parameters"""

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, parameters):
        reg_loss = None
        for W in parameters:
            if reg_loss is None:
                reg_loss = W.norm(2)
            else:
                reg_loss = reg_loss + W.norm(2)
        return reg_loss


class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


class EmbMarginLoss(nn.Module):
    """EmbMarginLoss, regularization on embeddings"""

    def __init__(self, power=2):
        super(EmbMarginLoss, self).__init__()
        self.power = power

    def forward(self, *embeddings):
        dev = embeddings[-1].device
        cache_one = torch.tensor(1.0).to(dev)
        cache_zero = torch.tensor(0.0).to(dev)
        emb_loss = torch.tensor(0.0).to(dev)
        for embedding in embeddings:
            norm_e = torch.sum(embedding**self.power, dim=1, keepdim=True)
            emb_loss += torch.sum(torch.max(norm_e - cache_one, cache_zero))
        return emb_loss


class InfoNCELoss(nn.Module):
    """InfoNCELoss, based on InfoNCE contrastive learning
    
    Implements the InfoNCE loss as described in the original paper:
    L_cl(s_u^ai, s_u^aj) = -log ( exp(sim(s_u^ai, s_u^aj)) / (exp(sim(s_u^ai, s_u^aj)) + sum_{s^- in S^-} exp(sim(s_u^ai, s^-))) )
    
    For each user, we have two augmented views (positive pair), and all other augmented 
    sequences in the batch serve as negative samples.
    """

    def __init__(self, temperature=0.07, similarity_type="cosine"):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.similarity_type = similarity_type

    def forward(self, z_i, z_j):
        """Compute InfoNCE loss between positive pairs and negative samples.
        
        Args:
            z_i (torch.tensor): First augmented view, shape (N, T, D) or (N, D)
            z_j (torch.tensor): Second augmented view, shape (N, T, D) or (N, D)
            
        Returns:
            loss (torch.tensor): InfoNCE loss scalar
        """
        # Handle different input shapes
        if z_i.dim() == 3:
            N, T, D = z_i.shape
            z_i = z_i.view(N * T, D)
            z_j = z_j.view(N * T, D)
            N = N * T
        else:
            N, D = z_i.shape
            
        # Normalize embeddings
        z_i = z_i / z_i.norm(dim=-1, keepdim=True)
        z_j = z_j / z_j.norm(dim=-1, keepdim=True)
        
        # Compute similarity matrix between all pairs
        if self.similarity_type == "cosine":
            # Compute cosine similarity between all pairs
            sim_matrix = torch.mm(z_i, z_j.t()) / self.temperature
        elif self.similarity_type == "dot":
            # Compute dot product similarity
            sim_matrix = torch.mm(z_i, z_j.t()) / self.temperature
        else:
            raise ValueError(f"Invalid similarity type: {self.similarity_type}")
        
        # Vectorized InfoNCE computation
        # Positive similarities are on the diagonal
        positive_sim = torch.diag(sim_matrix)  # Shape: (N,)
        
        # Create mask to exclude diagonal elements (positive pairs)
        mask = torch.eye(N, dtype=torch.bool, device=sim_matrix.device)
        
        # Compute log-sum-exp for negative samples (excluding diagonal)
        # Use logsumexp for numerical stability
        neg_sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
        log_sum_exp_neg = torch.logsumexp(neg_sim_matrix, dim=1)  # Shape: (N,)
        
        # Compute InfoNCE loss: -log(exp(pos_sim) / (exp(pos_sim) + exp(log_sum_exp_neg)))
        # This is equivalent to: pos_sim - torch.logaddexp(pos_sim, log_sum_exp_neg)
        loss = -positive_sim + torch.logaddexp(positive_sim, log_sum_exp_neg)
        
        return loss.mean()