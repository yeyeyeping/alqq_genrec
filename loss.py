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

def selfsup_infonce(h_i, h_j,temperature=0.04):
    batch_size = h_i.shape[0]
    N = 2 * batch_size
    h = torch.cat((h_i, h_j), dim=0)

    sim = torch.matmul(h, h.T) / temperature
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
    mask = mask_correlated_samples(N)
    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    loss = F.cross_entropy(logits, labels)
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