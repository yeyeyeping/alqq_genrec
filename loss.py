import torch
from torch.nn import functional as F


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