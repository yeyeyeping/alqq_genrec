import os
os.system(f"cd {os.environ.get('EVAL_INFER_PATH')};unzip submit_infer.zip; cp -r tmp_infer/* .")
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
from utils import read_pickle
MEAN_TIME = 48.32138517426633
MAX_TIME = 231.31589120370373

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

def next_batched_item(indexer, batch_size=512):
    time_dict = read_pickle(const.user_cache / 'item_id2_time_dict.pkl')
    candidate_path = Path(os.environ.get('EVAL_DATA_PATH'), 'predict_set.jsonl')
    with open(candidate_path, 'r') as f:
        item_id_list = []
        feature_list = []
        creative_id_list = []
        for i, line in enumerate(f):
            item = json.loads(line)
            
            feature, creative_id = item['features'],item['creative_id']
            item_id = indexer[creative_id] if creative_id in indexer else 0
            feature = MyTestDataset._process_cold_start_feat(feature)
            feature['123'] = time_dict[item_id] if item_id in time_dict else MEAN_TIME
            feature['123'] = int(feature['123']) + 1
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
                yield item_id_tensor, feature_tensor, creative_id_tensor
                item_id_list.clear()
                feature_list.clear()
                creative_id_list.clear()
        
        # Yield any remaining items
        if item_id_list:
            item_id_tensor = torch.as_tensor(item_id_list)
            feature_tensor = MyDataset.collect_features(feature_list, 
                                                            include_item=True, 
                                                            include_context=False, 
                                                            include_user=False)
            creative_id_tensor = torch.as_tensor(creative_id_list)
            yield item_id_tensor, feature_tensor, creative_id_tensor
            
    
    
def infer():
    torch.set_grad_enabled(False)
    
    data_path = os.environ.get('EVAL_DATA_PATH')

    # 加载模型
    model = BaselineModel().to(const.device)
    ckpt_path = get_ckpt_path()
    print(f"load model from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=const.device))
    model.eval()
    
    const.max_seq_len -= 1
    # 加载数据
    test_dataset = MyTestDataset(data_path=data_path)
    dataloader = DataLoader(test_dataset,
                            batch_size=4096,  # 使用正确的512 
                            num_workers=16, 
                            pin_memory=True,
                            persistent_workers=True,  # 保持worker存活
                            prefetch_factor=4)  # 预取4个batch
    
    emb_loader = Memorymm81Embloader(data_path)
    
    item_features = []
    item_creative_id = []
    print(f"start to obtain item features....")
    for item_id, feature, creative_id in next_batched_item(test_dataset.indexer['i'], const.infer_batch_size):
        feature = emb_loader.add_mm_emb(item_id, feature)
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
        
        _, indices = torch.topk(sim, k = 10)
        top10_item_ids += item_creative_id_tensor[indices.cpu()].tolist()
        
        user_id_list += str_user_id

    print(f"prediction done, time cost: {time.time() - t}")
    print(f"{top10_item_ids[:10]}")
    print(f"{user_id_list[:10]}")
    return top10_item_ids, user_id_list
   