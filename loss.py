import torch
from torch.nn import functional as F


def info_nce_loss(anchor_emb, pos_emb, neg_emb, temperature, return_logits=False):
    focal_loss = FocalLoss()
    device = anchor_emb.device
        
    true_logits = torch.sum(anchor_emb * pos_emb, dim=-1,keepdim=True)    
    neg_logits = anchor_emb @ neg_emb.T
    
    logits = torch.cat([true_logits, neg_logits], dim=1)
    
    label = torch.zeros(logits.shape[0], device=device,dtype=torch.long)                
    loss = focal_loss(logits / temperature, label)  
    
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


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha_spam: float = 0.2, alpha_ham: float = 0.8,
                 gamma: float = 3.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha_spam, alpha_ham], dtype=torch.float)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none',label_smoothing=0.1)
        pt = torch.exp(-ce_loss)                   # p_t
        focal_term = (1 - pt) ** self.gamma        # (1−p_t)^γ
        alpha_factor = self.alpha.to(inputs.device)[targets]
        loss = alpha_factor * focal_term * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss