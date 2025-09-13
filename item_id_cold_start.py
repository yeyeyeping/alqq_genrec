import os
os.system(f"cd {os.environ.get('EVAL_INFER_PATH')};unzip submit.zip; cp -r submit/* .")
import torch.nn.functional as F
import os
from pathlib import Path
from utils import read_pickle
import json
import numpy as np
import torch
import const
import time
import gc
import pickle
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
    
     
    cold_start_creative_id_list = []
    cold_start_item_embedding_list = []
    
    with open(candidate_path, 'r') as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            creative_id = item['creative_id']
            if creative_id not in indexer and creative_id in mm_emb_dict:
            # if  creative_id in mm_emb_dict:
                cold_start_creative_id_list.append(creative_id)
                cold_start_item_embedding_list.append(torch.as_tensor(mm_emb_dict[creative_id],dtype=torch.float32))
    print(f"creating top 20 similar item for {len(cold_start_creative_id_list)} cold start items")
    cold_start_item_embedding = torch.stack(cold_start_item_embedding_list, dim=0).to(const.device)
    cold_start_top_sim20 = []
    for st in range(0, len(cold_start_item_embedding), 256):
        end = min(st + 1024, len(cold_start_item_embedding))
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
    with open(cache_path/"data_info.pkl", "rb") as f:
            data_info = pickle.load(f)
    return data_info['item_expression_num']

def infer():
    top20similar_item()
    time.sleep(3000)
    
    
    
    