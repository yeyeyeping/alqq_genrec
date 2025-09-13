from operator import index
import os
os.system(f"cd {os.environ.get('EVAL_INFER_PATH')};unzip submit.zip; cp -r submit/* .")
import json
from pathlib import Path
from model import BaselineModel
from dataset import MyTestDataset,MyDataset
from torch.utils.data import DataLoader
import torch
import const
from  mm_emb_loader import Memorymm81Embloader
from torch.nn import functional as F
import time
import numpy as np
import gc
from utils import read_pickle
MEAN_TIME = 48.32138517426633
MAX_TIME = 231.31589120370373
def collect_all_creative_embedding(mm_emb_dict):
    all_ids = []
    all_embeddings = []
    for k, v in mm_emb_dict.items():
        if isinstance(v, np.ndarray):
            all_embeddings.append(torch.as_tensor(v, dtype=torch.float32))
            all_ids.append(k)
    return torch.stack(all_embeddings, dim=0), torch.as_tensor(all_ids, dtype=torch.long)
        
    
def top20similar_item():
    data_path = Path(os.environ.get('EVAL_DATA_PATH'))
    candidate_path = data_path/ 'predict_set.jsonl'
    indexer = read_pickle(data_path/ "indexer.pkl")['i']
    mm_emb_dict = read_pickle(data_path/"creative_emb" / "emb_81_32.pkl")
    all_creative_embedding, all_creative_id = collect_all_creative_embedding(mm_emb_dict)
    print(f"loadding {len(all_creative_embedding)} creative embeddings")
    
    all_creative_embedding = all_creative_embedding.to(const.device)
    all_creative_embedding = F.normalize(all_creative_embedding, dim=-1)
    
     
    cold_start_creative_id_list = []
    cold_start_item_embedding_list = []
    
    with open(candidate_path, 'r') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            creative_id = item['creative_id']
            if creative_id not in indexer and creative_id in mm_emb_dict:
                cold_start_creative_id_list.append(creative_id)
                cold_start_item_embedding_list.append(torch.as_tensor(mm_emb_dict[creative_id],dtype=torch.float32))
    print(f"creating top 20 similar item for {len(cold_start_creative_id_list)} cold start items")
    cold_start_item_embedding = torch.stack(cold_start_item_embedding_list, dim=0).to(const.device)
    cold_start_top_sim20 = []
    for st in range(0, len(cold_start_item_embedding), 256):
        end = min(st + 256, len(cold_start_item_embedding))
        src_emb = cold_start_item_embedding[st:end]
        src_emb = F.normalize(src_emb, dim=-1)
        with torch.amp.autocast(device_type=const.device, dtype=torch.bfloat16):
            sim = src_emb @ all_creative_embedding.T
        _, indices = torch.topk(sim, k=10)
        
        
        sim_create_id = all_creative_id[indices.cpu()].cpu().tolist()
        cold_start_top_sim20 += [[indexer[i] for i in topsim20 if i in indexer] for topsim20 in sim_create_id]
    print(f"cold_start_creative_id_list: {cold_start_creative_id_list[:10]}")
    print(f"sim_create_id: {cold_start_top_sim20[:10]}")
    
    del indexer, all_creative_embedding, all_creative_id,mm_emb_dict,cold_start_item_embedding,cold_start_item_embedding_list
    gc.collect()
    return dict(zip(cold_start_creative_id_list, cold_start_top_sim20))

def read_item_expression_dict():
    cache_path = Path(os.environ.get('USER_CACHE_PATH'))
    data_info = read_pickle(cache_path/"data_info.pkl")
    return data_info['item_expression_num'],data_info['item_click_num']
def get_ckpt_path():
    ckpt_path = os.environ.get("MODEL_OUTPUT_PATH")
    if ckpt_path is None:
        raise ValueError("MODEL_OUTPUT_PATH is not set")
    for item in os.listdir(ckpt_path):
        if item.endswith(".pt"):
            return os.path.join(ckpt_path, item)
def to_device(batch):
    for k,v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(const.device, non_blocking=True)
    return batch

def next_batched_item(mm_emb_dict, top20similar_item_dict, indexer, batch_size=512):
    time_dict = read_pickle(const.user_cache / 'item_id2_time_dict.pkl')
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    with open(candidate_path, 'r') as f:
        item_id_list = []
        feature_list = []
        creative_id_list = []
        mm_feat_emb = []
        
        for i, line in enumerate(f):
            item = json.loads(line)
            
            feature, creative_id = item['features'],item['creative_id']
            feature = MyTestDataset._process_cold_start_feat(feature)
            #冷启动物品
            if creative_id not in indexer:
                # 找到最相似的物品，id赋予他
                item_id = 0
                sim_item_list = top20similar_item_dict.get(creative_id, [0, ])
                if len(sim_item_list) != 0:
                    item_id = sim_item_list[0]
                        
                if "123" in const.item_feature.sparse_feature_ids:
                    feature['123'] = MEAN_TIME
                    feature['123'] = int(feature['123']) + 1
            else:
                item_id = indexer[creative_id]
                if "123" in const.item_feature.sparse_feature_ids:
                    feature['123'] = time_dict[item_id] if item_id in time_dict else MEAN_TIME
                    feature['123'] = int(feature['123']) + 1
            
            if creative_id in mm_emb_dict:
                mm_feat_emb.append(torch.as_tensor(mm_emb_dict[creative_id],dtype=torch.float32))    
            else:
                mm_feat_emb.append(torch.zeros(32,dtype=torch.float32))
            
            item_id_list.append(item_id)
            feature_list.append(feature)
            creative_id_list.append(creative_id)
            
            # Yield when we have accumulated batch_size items
            if len(item_id_list) == batch_size:
                item_id_tensor = torch.as_tensor(item_id_list)
                feature_tensor = MyDataset.collect_features(feature_list, 
                                                            include_item=True, 
                                                            include_context=False, 
                                                            include_user=False)
                creative_id_tensor = torch.as_tensor(creative_id_list)
                mm_feat_emb_tensor = torch.stack(mm_feat_emb, dim=0)
                feature_tensor['81'] = mm_feat_emb_tensor
                yield item_id_tensor, feature_tensor, creative_id_tensor
                item_id_list.clear()
                feature_list.clear()
                creative_id_list.clear()
                mm_feat_emb.clear()
        
        # Yield any remaining items
        if item_id_list:
            item_id_tensor = torch.as_tensor(item_id_list)
            feature_tensor = MyDataset.collect_features(feature_list, 
                                                            include_item=True, 
                                                            include_context=False, 
                                                            include_user=False)
            creative_id_tensor = torch.as_tensor(creative_id_list)
            mm_feat_emb_tensor = torch.stack(mm_feat_emb, dim=0)
            feature_tensor['81'] = mm_feat_emb_tensor
            yield item_id_tensor, feature_tensor, creative_id_tensor
            
    
    
def infer():
    torch.set_grad_enabled(False)
    
    data_path = Path(os.environ.get('EVAL_DATA_PATH'))
    mm_emb_dict = read_pickle(data_path/"creative_emb" / "emb_81_32.pkl")
    top20similar_item_dict = top20similar_item()
    item_expression_dict, item_click_dict = read_item_expression_dict()
    # 加载模型
    model = BaselineModel().to(const.device)
    ckpt_path = get_ckpt_path()
    print(f"load model from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=const.device))
    model.eval()
    
    const.max_seq_len -= 1
    # 加载数据
    time_dict = read_pickle(const.user_cache / 'item_id2_time_dict.pkl')
    test_dataset = MyTestDataset(data_path, time_dict)
    dataloader = DataLoader(test_dataset,
                            batch_size=1024,  # 使用正确的512 
                            num_workers=16, 
                            pin_memory=True,
                            persistent_workers=True,  # 保持worker存活
                            prefetch_factor=4)  # 预取4个batch
    
    emb_loader = Memorymm81Embloader(data_path)
    
    item_features = []
    item_creative_id = []
    print(f"start to obtain item features....")
    for item_id, feature, creative_id in next_batched_item(mm_emb_dict, top20similar_item_dict, test_dataset.indexer['i'], const.infer_batch_size):
        # feature = emb_loader.add_mm_emb(item_id, feature)
        item_id = item_id.to(const.device)
        feature = to_device(feature)
        with torch.amp.autocast(device_type=const.device, dtype=torch.bfloat16):
            item_emb = F.normalize(model.item_tower(item_id, feature), dim=-1)
        item_features.append(item_emb)
        item_creative_id.append(creative_id)
    print(f"loadding {len(item_features)} item features")
    item_features_tensor = torch.cat(item_features, dim=0).to(const.device)
    item_creative_id_tensor = torch.cat(item_creative_id, dim=0)
    print(f"loadding {len(item_creative_id_tensor)} item features")
    print(f"item feature: {item_features_tensor[:10]}")
    print(f"creative id: {item_creative_id_tensor[:10]}")    
    

    user_id_list = []
    top10_item_ids = []
    top10_sim_scores = []
    t = time.time()
    print(f"start to predict {len(dataloader) * dataloader.batch_size} user seqs")
    for str_user_id, user_id, user_feat, action_type, item_id, item_feat, context_feat in dataloader:
        item_feat = emb_loader.add_mm_emb(item_id, item_feat)
        user_id,item_id = user_id.to(const.device), item_id.to(const.device)
        user_feat, item_feat,context_feat = to_device(user_feat), to_device(item_feat), to_device(context_feat)
        
        with torch.amp.autocast(device_type=const.device, dtype=torch.bfloat16):
            next_token_emb = model(user_id, user_feat,item_id, item_feat, context_feat)
            next_token_emb = F.normalize(next_token_emb[:,-1,:], dim=-1)
            sim = next_token_emb @ item_features_tensor.T
        
        top10_sim, indices = torch.topk(sim, k = 10)
        top10_sim= top10_sim / 2 + 0.5
        top10_item_ids += item_creative_id_tensor[indices.cpu()].tolist()
        top10_sim_scores += top10_sim.cpu().tolist()
        user_id_list += str_user_id
    gc.collect()
    print(f"prediction done, time cost: {time.time() - t}")
    print(f"{top10_item_ids[:10]}")
    print(f"{top10_sim_scores[:10]}")
    print(f"{user_id_list[:10]}")
    
    print("start to sort top10 item according to expression")
    
    t = time.time()
    for i, (sim_score, items) in enumerate(zip(top10_sim_scores, top10_item_ids)):
        expr = [s + (item_click_dict.get(x, 0)/item_expression_dict.get(x, 1)) for (s, x) in zip(sim_score, items)]
        sorted_ids, _ = zip(*sorted(zip(items, expr), key=lambda x: x[1], reverse=True))
        top10_item_ids[i] = list(sorted_ids)
        
    print(f"sort done, time cost: {time.time() - t}")
    
    return top10_item_ids, user_id_list
    