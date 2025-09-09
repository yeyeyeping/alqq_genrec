import torch
from torch.nn import functional as F

def mask_correlated_samples( N):
    mask = torch.ones((N, N))
    mask = mask.fill_diagonal_(0)
    for i in range(N//2):
        mask[i, N//2 + i] = 0
        mask[N//2 + i, i] = 0
    mask = mask.bool()
    return mask

def selfsup_infonce(h_i, h_j,temperature=0.04,num_neg=256):
    pos_sim = torch.sum(h_i * h_j, dim=-1,keepdim=True)
    # 随机挑选256作为负样本（每个样本各自随机采样，且不包含自身索引）
    device = h_i.device
    N = h_i.shape[0]
    # 为每一行构造 K 个不含自身索引的负样本索引
    # 技巧：先从 [0, N-2] 采样，然后对 >= 行号 的元素加 1 来跳过自身索引
    row_ids = torch.arange(N, device=device).unsqueeze(1).expand(N, num_neg)
    neg_idx = torch.randint(0, N - 1, (N, num_neg), device=device)
    neg_idx = neg_idx + (neg_idx >= row_ids).long()

    neg_vec = h_j[neg_idx]
    neg_sim = (h_i.unsqueeze(1) * neg_vec).sum(dim=-1)

    logits = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(N, device=device, dtype=torch.long)
    loss = F.cross_entropy(logits / temperature, labels)
    return loss

def info_nce_loss(anchor_emb, pos_emb, neg_emb, temperature, return_logits=False):
    device = anchor_emb.device
        
    true_logits = torch.sum(anchor_emb * pos_emb, dim=-1,keepdim=True)    
    neg_logits = anchor_emb @ neg_emb.T
    
    logits = torch.cat([true_logits, neg_logits], dim=1)
    
    label = torch.zeros(logits.shape[0], device=device,dtype=torch.long)                
    loss = F.cross_entropy(logits / temperature, label)  
    
    if return_logits:
        return loss, neg_logits.mean().item(), true_logits.mean().item(), logits
    else:
        return loss, neg_logits.mean().item(), true_logits.mean().item()

def l2_reg_loss(model,l2_alpha):
    loss = 0.0
    for param in model.item_tower.sparse_emb['item_id'].parameters():
        loss += l2_alpha * torch.norm(param)
            
    for param in model.user_tower.sparse_emb['user_id'].parameters():
        loss += l2_alpha * torch.norm(param)
    return loss