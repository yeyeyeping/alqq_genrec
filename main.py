from sampler import sample_neg
import json
import os
import time
from pathlib import Path
import const
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import gc
from dataset import MyDataset
from model.model import BaselineModel
from torch.amp import autocast, GradScaler
from timm.scheduler import CosineLRScheduler
from torch.nn import functional as F
from utils import seed_everything, seed_worker
from loss import info_nce_loss,l2_reg_loss
from mm_emb_loader import Memorymm81Embloader
from torch.optim import SGD
def build_dataloader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        prefetch_factor= 4,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
    )

def apply_model_init(model:BaselineModel):
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except Exception:
            pass
    for k in model.user_tower.sparse_emb:
        model.user_tower.sparse_emb[k].weight.data[0, :] = 0
    
    for k in model.item_tower.sparse_emb:
        model.item_tower.sparse_emb[k].weight.data[0, :] = 0
    
    model.pos_embedding.weight.data[0, :] = 0


def to_device(batch):
    for k,v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(const.device, non_blocking=True)
    return batch

def make_input_and_label(seq_id, token_type, action_type, feat):
    input_ids = seq_id[:,:-1]
    input_token_type = token_type[:,:-1]
    input_action_type = action_type[:,:-1]
    input_feat = {k:v[:,:-1] for k,v in feat.items()}
    
    
    label_ids = seq_id[:,1:].clone()
    label_token_type = token_type[:,1:].clone()
    label_action_type = action_type[:,1:].clone()
    label_feat = {k:v[:,1:].clone() for k,v in feat.items()}
    
    return input_ids, input_token_type, input_action_type, input_feat, label_ids, label_token_type, label_action_type, label_feat

def train_one_step(batch, emb_loader, loader, model:BaselineModel):
    
    seq_id, token_type, action_type, feat = batch
    feat = emb_loader.add_mm_emb(seq_id, feat, token_type == 1)

    # 负样本采样
    neg_id, neg_feat = next(loader)
    neg_feat = emb_loader.add_mm_emb(neg_id, neg_feat)
    seq_id, token_type, action_type, feat = \
                seq_id.to(const.device,non_blocking=True), \
                token_type.to(const.device,non_blocking=True), \
                action_type.to(const.device,non_blocking=True), \
                to_device(feat)
        
    neg_id, neg_feat = neg_id.to(const.device, non_blocking=True), to_device(neg_feat)
    
    with autocast(device_type=const.device, dtype=torch.bfloat16):        
        input_ids, input_token_type, input_action_type, input_feat, next_ids, next_token_type, next_action_type, next_feat \
                    = make_input_and_label(seq_id, token_type, action_type, feat)
        
        
        next_token_emb = model(input_ids, input_token_type, input_feat)
        
        neg_emb = model.forward_item(neg_id, neg_feat)
        pos_emb = model.forward_item(next_ids, next_feat, next_token_type)
        
        indices = torch.where(next_token_type == 1) 
        mask = torch.isin(neg_id,seq_id,)
        
        anchor_emb = F.normalize(next_token_emb[indices[0],indices[1],:],dim=-1)
        pos_emb = F.normalize(pos_emb[indices[0],indices[1],:],dim=-1)
        neg_emb = F.normalize(neg_emb[~mask], dim=-1)
        loss, neg_sim, pos_sim = info_nce_loss(anchor_emb, pos_emb, neg_emb, const.temperature)
        loss += l2_reg_loss(model,const.l2_alpha)
    return loss, neg_sim, pos_sim

@torch.no_grad()
def valid_one_step(batch, emb_loader, loader, model:BaselineModel):
    seq_id, token_type, action_type, feat = batch
    feat = emb_loader.add_mm_emb(seq_id, feat, token_type == 1)

    # 负样本采样
    neg_id, neg_feat = next(loader)
    neg_feat = emb_loader.add_mm_emb(neg_id, neg_feat)
    seq_id, token_type, action_type, feat = \
                seq_id.to(const.device,non_blocking=True), \
                token_type.to(const.device,non_blocking=True), \
                action_type.to(const.device,non_blocking=True), \
                to_device(feat)
        
    neg_id, neg_feat = neg_id.to(const.device, non_blocking=True), to_device(neg_feat)
    
    with autocast(device_type=const.device, dtype=torch.bfloat16):        
        input_ids, input_token_type, input_action_type, input_feat, next_ids, next_token_type, next_action_type, next_feat \
                    = make_input_and_label(seq_id, token_type, action_type, feat)
        
        
        next_token_emb = model(input_ids, input_token_type, input_feat)
        
        neg_emb = model.forward_item(neg_id, neg_feat)
        pos_emb = model.forward_item(next_ids, next_feat, next_token_type)
        
        indices = torch.where(next_action_type == 1) 
        mask = torch.isin(neg_id,seq_id,)
    
        
        anchor_emb = F.normalize(next_token_emb[indices[0],indices[1],:],dim=-1)
        pos_emb = F.normalize(pos_emb[indices[0],indices[1],:],dim=-1)
        neg_emb = F.normalize(neg_emb[~mask], dim=-1)[:30000]
        loss, neg_sim, pos_sim, logits = info_nce_loss(anchor_emb, pos_emb, neg_emb, const.temperature,return_logits=True)
        
        
        _, top1_indices = torch.topk(logits, k=1, largest=True, dim=1)
        _, top10_indices = torch.topk(logits, k=10, largest=True, dim=1)
        
        top1_correct = torch.any(top1_indices == 0, dim=1).sum().item()
        top10_correct = torch.any(top10_indices == 0, dim=1).sum().item()
        
    return loss.item(), neg_sim, pos_sim, top1_correct, top10_correct,logits.shape[0]
    
if __name__ == '__main__':
    Path(os.environ.get('TRAIN_LOG_PATH')).mkdir(parents=True, exist_ok=True)
    Path(os.environ.get('TRAIN_TF_EVENTS_PATH')).mkdir(parents=True, exist_ok=True)
    log_file = open(Path(os.environ.get('TRAIN_LOG_PATH'), 'train.log'), 'w')
    writer = SummaryWriter(os.environ.get('TRAIN_TF_EVENTS_PATH'))
    # global dataset
    seed_everything(const.seed)
    
    dataset = MyDataset(const.data_path)
    # train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = build_dataloader(dataset, const.batch_size, const.num_workers, True)
    # valid_loader = build_dataloader(valid_dataset, const.batch_size, const.num_workers, False)
    
    
    model = BaselineModel().to(const.device)
    print(model)
    
    apply_model_init(model)


    optimizer = torch.optim.AdamW(model.parameters(), lr=const.lr, betas=(0.9, 0.99))
    scheduler = CosineLRScheduler(
                        optimizer, 
                        t_initial=const.num_epochs * len(train_loader) - 4000,  
                        warmup_t=const.warmup_t, 
                        lr_min=5e-5, 
                        warmup_lr_init=1e-5, 
                        t_in_epochs=False)
    
    scaler = GradScaler()
    
    global_step = 0
    total_step = const.num_epochs * len(train_loader)
    neg_loader = iter(sample_neg())
    emb_loader = Memorymm81Embloader(const.data_path)
    print("Start training")
    
    for epoch in range(1, const.num_epochs + 1):
        model.train()

        for step, batch in enumerate(train_loader):
            st_time = time.perf_counter()
            optimizer.zero_grad()
            
            loss, neg_sim, pos_sim = train_one_step(batch, emb_loader, neg_loader, model)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), const.grad_norm).item()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step_update(global_step)
            
            loss = loss.item()
            
            log_json = json.dumps(
                {
                    'global_step': f"{global_step}/{total_step}", 
                    "grad_norm":grad_norm,
                    'loss': loss, 
                    "sim_pos":pos_sim,
                    "sim_neg":neg_sim,
                    'epoch': epoch, 
                    'lr': optimizer.param_groups[0]['lr'],
                    'time': time.perf_counter() - st_time
                }
            )
            
            st_time = time.perf_counter()
            
            log_file.write(log_json + '\n')
            log_file.flush()
            
            print("[TRAIN] " + log_json)
            writer.add_scalar('train/sim_pos', pos_sim, global_step)
            writer.add_scalar('train/sim_neg', neg_sim, global_step)
            writer.add_scalar('train/sim_gap', pos_sim - neg_sim, global_step)
            writer.add_scalar('train/loss', loss, global_step)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('train/grad_norm', grad_norm, global_step)
            global_step += 1
        save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"epoch={epoch}_global_step={global_step}.training_loss={loss:.4f}")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(model.state_dict(), save_dir / "model.pt")
        # model.eval()
        # valid_loss_sum = 0
        # valid_top1_acc_sum = 0
        # valid_top10_acc_sum = 0
        # valid_sim_pos_sum = 0
        # valid_sim_neg_sum = 0
        # valid_sim_gap_sum = 0
        # total_sample = 0
        # for step, batch in enumerate(valid_loader):
        #     loss, neg_sim, pos_sim, top1_correct, top10_correct, num_sample = valid_one_step(batch, emb_loader, neg_loader, model)
            
        #     valid_loss_sum += loss * num_sample
        #     valid_top1_acc_sum += top1_correct
        #     valid_top10_acc_sum += top10_correct
        #     total_sample += num_sample
        #     valid_sim_pos_sum += pos_sim * num_sample
        #     valid_sim_neg_sum += neg_sim * num_sample
        #     valid_sim_gap_sum += (pos_sim - neg_sim) * num_sample
        
        # valid_loss = valid_loss_sum / total_sample
        # valid_top1_acc = valid_top1_acc_sum / total_sample
        # valid_top10_acc = valid_top10_acc_sum / total_sample
        # valid_sim_pos = valid_sim_pos_sum / total_sample
        # valid_sim_neg = valid_sim_neg_sum / total_sample
        # valid_sim_gap = valid_sim_gap_sum / total_sample
        
        # print(f"[EVAL] loss: {valid_loss:.4f}, top1 acc: {valid_top1_acc:.4f}, top10 acc: {valid_top10_acc:.4f}, sim_pos: {valid_sim_pos:.4f}, sim_neg: {valid_sim_neg:.4f}, sim_gap: {valid_sim_gap:.4f}")
        
        # writer.add_scalar('valid/loss', valid_loss, global_step)
        # writer.add_scalar('valid/top1_acc', valid_top1_acc, global_step)
        # writer.add_scalar('valid/top10_acc', valid_top10_acc, global_step)
        # writer.add_scalar('valid/sim_pos', valid_sim_pos, global_step)
        # writer.add_scalar('valid/sim_neg', valid_sim_neg, global_step)
        # writer.add_scalar('valid/sim_gap', valid_sim_gap, global_step)

        # save_dir = Path(os.environ.get('TRAIN_CKPT_PATH'), f"global_step{global_step}.loss={valid_loss:.4f}.acc_top1={valid_top1_acc:.4f}.acc_top10={valid_top10_acc:.4f}")
        # save_dir.mkdir(parents=True, exist_ok=True)
        
        # torch.save(model.state_dict(), save_dir / "model.pt")
        
        gc.collect()
    print("Done")
    print(const.__dict__)
    writer.close()
    log_file.close()
