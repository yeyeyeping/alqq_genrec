from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from utils import read_pickle
import json
import const
from collections import defaultdict
import torch
from datetime import datetime
import pandas as pd

# from torch.utils.data._utils.collate import default_collate
# import numpy as np
# from sampler import BaseSampler
MIN_TS = 1728921670
MAX_TS = 1748907455
class MyDataset(Dataset):
    def __init__(self, data_path): 
        super().__init__()
        self.data_path = data_path
        self.seq_offsets = read_pickle(Path(data_path, 'seq_offsets.pkl'))
        self.seq_file_fp = None
    def __len__(self):
        return len(self.seq_offsets)
    
    def _load_user_data(self, uid):
        # uid时reid后的结果
        if self.seq_file_fp is None:
            self.seq_file_fp = open(Path(self.data_path, 'seq.jsonl'), 'rb')
        self.seq_file_fp.seek(self.seq_offsets[uid])
        line = self.seq_file_fp.readline()
        data = json.loads(line)
        return data

    def format_user_seq(self, user_sequence):
        user_sequence = sorted(user_sequence, key=lambda x: x[-1])
        ext_user_sequence = []
        uid, ufeat = 0, {}
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, ts = record_tuple
            if u and user_feat:
                uid = u
                ufeat = user_feat
                    
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, action_type, ts))
        return uid, ufeat, ext_user_sequence
    

    def add_time_feat(self, ts_array):
        ts_tensor = torch.as_tensor(ts_array)
        time_gap = torch.diff(ts_tensor, prepend=ts_tensor[0][None])
        log_gap = torch.log1p(time_gap).int() + 1
        log_gap = torch.clamp(log_gap, max=const.model_param.max_diff)
        
        ts_utc8 = ts_tensor + 8 * 3600
        dt = pd.to_datetime(ts_utc8, unit='s')
        hours = torch.as_tensor(dt.hour) + 1
        weekdays = torch.as_tensor(dt.weekday) + 1
        months = torch.as_tensor(dt.month) + 1
        days = torch.as_tensor(dt.day) + 1

        decay = (torch.as_tensor(MAX_TS) - ts_tensor) / 86400 
        delta_scaled = torch.log1p(decay).int() + 1
        
        delta_scaled = torch.clamp(delta_scaled, max=const.model_param.max_decay)
        out_time_feat = {
            "201": MyDataset.pad_seq(log_gap, const.max_seq_len, 0),
            "202": MyDataset.pad_seq(weekdays, const.max_seq_len, 0),
            "203": MyDataset.pad_seq(hours, const.max_seq_len, 0),
            "204": MyDataset.pad_seq(months, const.max_seq_len, 0),
            "205": MyDataset.pad_seq(days, const.max_seq_len, 0),
            "206": MyDataset.pad_seq(delta_scaled, const.max_seq_len, 0)
            
        }
        return out_time_feat
    
    @classmethod
    def ensure_user_feat(cls, feat):
        filled_feat = {}
        for feat_id in const.user_feature.all_feature_ids:
                if feat_id not in feat.keys():
                    filled_feat[feat_id] = torch.as_tensor(const.user_feature.fill(feat_id))
                else:
                    if feat_id in const.user_feature.array_feature_ids:
                        filled_feat[feat_id] = torch.as_tensor(const.user_feature.pad_array_feature(feat_id, feat[feat_id]))
                    else:
                        filled_feat[feat_id] = torch.as_tensor(feat[feat_id])
        return filled_feat
    
    @classmethod
    def fill_feature(cls, 
                     feat,
                     include_user=True, 
                     include_item=True,
                     include_context=True):
        filled_feat = {}
        if include_user:
            for feat_id in const.user_feature.all_feature_ids:
                if feat_id not in feat.keys():
                    filled_feat[feat_id] = const.user_feature.fill(feat_id)
                
                else:
                    if feat_id in const.user_feature.array_feature_ids:
                        filled_feat[feat_id] = const.user_feature.pad_array_feature(feat_id, feat[feat_id])
                    else:
                        filled_feat[feat_id] = feat[feat_id]
        
        if include_item:
            for feat_id in const.item_feature.all_feature_ids:
                if feat_id not in feat.keys():
                    filled_feat[feat_id] = const.item_feature.fill(feat_id)
                else:
                    filled_feat[feat_id] = feat[feat_id]
                    
        if include_context:
            for feat_id in const.context_feature.all_feature_ids:
                if feat_id not in feat.keys():
                    filled_feat[feat_id] = const.context_feature.fill(feat_id)
                else:
                    filled_feat[feat_id] = feat[feat_id]
        return filled_feat
    
    @classmethod
    def collect_features(cls, 
                         feat_list, 
                         include_user=True, 
                         include_item=True, 
                         include_context=True
                         ):
        feat_list = [MyDataset.fill_feature(feat, include_user, include_item, include_context) for feat in feat_list]
        feature_name_value_list = defaultdict(list)
        for feat in feat_list:
            for feat_id, feat_value in feat.items():
                feature_name_value_list[feat_id].append(feat_value)
        out_dict = {}
        
        for k, v in feature_name_value_list.items():
            if k in const.user_feature.array_feature_ids + const.user_feature.sparse_feature_ids  + const.item_feature.sparse_feature_ids + const.context_feature.sparse_feature_ids:
                out_dict[k] = torch.as_tensor(v, dtype=torch.int32)
            elif k in const.user_feature.dense_feature_ids + const.item_feature.dense_feature_ids:
                out_dict[k] = torch.as_tensor(v, dtype=torch.float32)
            else:
                print(f"Invalid feature id: {k}")
                        
        return out_dict
    
    @classmethod
    def pad_seq(cls, seq, seq_len, pad_value):
        pad_len = seq_len - len(seq)
        if isinstance(seq, list):
            if pad_len <= 0:
                return seq[:seq_len]
            else:
                pad_value = [pad_value, ] * pad_len
        elif isinstance(seq, torch.Tensor):
            if pad_len <= 0:
                return seq[:seq_len]
            else:
                pad_value = torch.as_tensor([pad_value, ]* pad_len, dtype=seq.dtype)
                return torch.cat([pad_value, seq], dim=0)
        else:
            raise ValueError(f"Invalid sequence type: {type(seq)}")
        
        return pad_value + seq
        
    def __getitem__(self, index):
        user_seq = self._load_user_data(index)
        user_id, user_feat, ext_user_seq = self.format_user_seq(user_seq)
        
        item_id_list = []
        action_type_list = []
        feat_list = []
        ts_list = []
        front_click_item = set()
        seq_list = []
        for i, feat, action_type, ts in ext_user_seq:
            item_id_list.append(i)
            action_type_list.append(action_type if action_type is not None else 0)
            feat_list.append(feat)
            
            clicked_item_list = list(front_click_item)
            click_seq = MyDataset.pad_seq(clicked_item_list[-const.context_feature.seq_len:].copy(), 
                                          const.context_feature.seq_len, 
                                          0)
                                        
            seq_list.append(click_seq)
            
            if action_type == 1:
                front_click_item.add(i)
                
            ts_list.append(ts)
            
        user_feat = MyDataset.ensure_user_feat(user_feat)
        item_id_list = MyDataset.pad_seq(item_id_list, const.max_seq_len, 0)
        action_type_list = MyDataset.pad_seq(action_type_list, const.max_seq_len, 0)
        seq_list = MyDataset.pad_seq(seq_list, const.max_seq_len, [0,]*const.context_feature.seq_len)
        feat_list = MyDataset.pad_seq(feat_list, const.max_seq_len, {})
        
        item_feat_dict = MyDataset.collect_features(feat_list,
                                                    include_item=True, 
                                                    include_context=False, 
                                                    include_user=False)
                
        time_feat = self.add_time_feat(ts_list)
        context_feat = {
            ** time_feat,
            "210": torch.as_tensor(seq_list, dtype=torch.int32)
        }
        
        return torch.as_tensor(user_id,dtype=torch.int32), user_feat,\
            torch.as_tensor(action_type_list, dtype=torch.bool), \
            torch.as_tensor(item_id_list, dtype=torch.int32), item_feat_dict,\
                context_feat

        
class MyTestDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = Path(data_path)
        self.seq_offsets = read_pickle(self.data_path/ 'predict_seq_offsets.pkl')
        self.indexer = read_pickle(self.data_path / 'indexer.pkl')
        self.indexer_u_rev = {v: k for k, v in self.indexer['u'].items()}
        
        self.itemnum = len(self.indexer['i'])
        self.seq_file_fp = None
        
    def __len__(self):
        return len(self.seq_offsets)
    
    def _load_user_data(self, uid):
        # uid时reid后的结果
        if self.seq_file_fp is None:
            self.seq_file_fp = open(self.data_path / 'predict_seq.jsonl', 'rb')
        self.seq_file_fp.seek(self.seq_offsets[uid])
        line = self.seq_file_fp.readline()
        data = json.loads(line)
        return data
    
    
    @classmethod
    def _process_cold_start_feat(cls, feat):
        """
        处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
        """
        processed_feat = {}
        for feat_id, feat_value in feat.items():
            if isinstance(feat_value, list):
                feat_value = [0 if isinstance(v, str) else v for v in feat_value]
            elif isinstance(feat_value, str):
                feat_value = 0
            else:
                processed_feat[feat_id] = feat_value
            
        return processed_feat
    
    def format_user_seq(self, user_sequence):
        user_sequence = sorted(user_sequence, key=lambda x: x[-1])
        ext_user_sequence = []
        user_id = None
        ureid = 0
        ufeat = {}
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, ts = record_tuple
            if u and user_id is None:
                if type(u) == str:  # 如果是字符串，说明是user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]
            
            if u and user_feat:
                if type(u) == str:
                    u = 0
                if user_feat:
                    user_feat = self._process_cold_start_feat(user_feat)
                ureid = u
                ufeat = user_feat
            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat,action_type, ts))
            
        return ext_user_sequence, user_id, ureid, ufeat

    def add_time_feat(self, ts_array):
        ts_tensor = torch.as_tensor(ts_array)
        time_gap = torch.diff(ts_tensor, prepend=ts_tensor[0][None])
        log_gap = torch.log1p(time_gap).int() + 1
        log_gap = torch.clamp(log_gap, max=const.model_param.max_diff)
        
        ts_utc8 = ts_tensor + 8 * 3600
        dt = pd.to_datetime(ts_utc8, unit='s')
        hours = torch.as_tensor(dt.hour) + 1
        weekdays = torch.as_tensor(dt.weekday) + 1
        months = torch.as_tensor(dt.month) + 1
        days = torch.as_tensor(dt.day) + 1

        decay = (torch.as_tensor(MAX_TS) - ts_tensor) / 86400 
        delta_scaled = torch.log1p(decay).int() + 1
        
        delta_scaled = torch.clamp(delta_scaled, max=const.model_param.max_decay)
        out_time_feat = {
            "201": MyDataset.pad_seq(log_gap, const.max_seq_len, 0),
            "202": MyDataset.pad_seq(weekdays, const.max_seq_len, 0),
            "203": MyDataset.pad_seq(hours, const.max_seq_len, 0),
            "204": MyDataset.pad_seq(months, const.max_seq_len, 0),
            "205": MyDataset.pad_seq(days, const.max_seq_len, 0),
            "206": MyDataset.pad_seq(delta_scaled, const.max_seq_len, 0)
            
        }
        return out_time_feat
    
    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)
        ext_user_sequence, str_user_id, user_id, user_feat = self.format_user_seq(user_sequence)
        
        item_id_list = []
        action_type_list = []
        feat_list = []
        ts_list = []
        front_click_item = set()
        seq_list = []
        for i, feat, action_type, ts in ext_user_sequence:
            item_id_list.append(i)
            action_type_list.append(action_type if action_type is not None else 0)
            feat_list.append(feat)
            
            clicked_item_list = list(front_click_item)
            click_seq = MyDataset.pad_seq(clicked_item_list[-const.context_feature.seq_len:].copy(), 
                                          const.context_feature.seq_len, 
                                          0)
            seq_list.append(click_seq)

            if action_type == 1:
                front_click_item.add(i)
                
            ts_list.append(ts)
            
            
        user_feat = MyDataset.ensure_user_feat(user_feat)
        item_id_list = MyDataset.pad_seq(item_id_list, const.max_seq_len, 0)
        action_type_list = MyDataset.pad_seq(action_type_list, const.max_seq_len, 0)
        seq_list = MyDataset.pad_seq(seq_list, const.max_seq_len , [0,]*const.context_feature.seq_len)
        feat_list = MyDataset.pad_seq(feat_list, const.max_seq_len, {})
        
        item_feat_dict = MyDataset.collect_features(feat_list,
                                                    include_item=True, 
                                                    include_context=False, 
                                                    include_user=False)                
        time_feat = self.add_time_feat(ts_list)
        context_feat = {
            ** time_feat,
            "210": torch.as_tensor(seq_list, dtype=torch.int32)
        }
        
        return str_user_id, torch.as_tensor(user_id,dtype=torch.int32), user_feat,\
            torch.as_tensor(action_type_list, dtype=torch.bool), \
            torch.as_tensor(item_id_list, dtype=torch.int32), item_feat_dict,\
                context_feat
    
if __name__ == "__main__":
    dataset = MyDataset(data_path='/home/yeep/project/alqq_generc/data/TencentGR_1k')
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    for d in dataloader:
        breakpoint()
        user_id, user_feat, action_type, item_id, item_feat, context_feat = d 


# if __name__ == "__main__":
#     from torch.utils.data import DataLoader
#     import dataset2
#     from mm_emb_loader import Memorymm81Embloader
#     dataset = MyDataset(data_path='/home/yeep/project/alqq_generc/data/TencentGR_1k')
#     dataloader = DataLoader(dataset, batch_size=77, )
    
    
#     ds2 = dataset2.MyDataset(data_dir='/home/yeep/project/alqq_generc/data/TencentGR_1k')
#     dataloader2 = DataLoader(ds2, batch_size=77, collate_fn=dataset2.MyDataset.collate_fn)
#     emb_loader = Memorymm81Embloader(data_path='/home/yeep/project/alqq_generc/data/TencentGR_1k')
#     for d1, d2 in zip(dataloader, dataloader2):
        
#         ids = d1["id"]
#         seq1, pos = ids[:,:-1],ids[:,1:].clone()
        
#         token_type1 = d1["token_type"]
        
#         seq2, pos2, _, token_type2, _, _, seq_feat, _, _ = d2
        
        
        # 验证sparse特征是否一致
        # boolsparse = True
        # for key in const.item_feature.sparse_feature_ids + const.user_feature.sparse_feature_ids:
        #     f101_1 = d1[key][:,:-1].reshape(-1)
        #     f101_2 = torch.as_tensor([i[key]  for seq in seq_feat for i in seq[1:]])
        #     boolsparse = boolsparse & ((f101_1 == f101_2).all())
        
        # # 验证用户的array特征
        
        # boolarray =  (torch.as_tensor([i['107']  for seq in seq_feat for i in seq[1:]])[:,0] == d1['107'][:,:-1].reshape(-1)).all()
        # print(boolarray)
        
        # print(boolsparse)
        
        # pos[token_type1[:,1:] != 1] = 0
        # print((seq1 == seq2[:, 1:]).all(),(pos == pos2[:, 1:]).all(),(token_type1[:,:-1] == token_type2[:,1:]).all())
        
        # 测试embedding加载是否一致
        # item_mask = (token_type1[:,:-1] == 1)
        
        # embeddings = emb_loader.batch_load_emb((seq1 * (item_mask == 1)).reshape(-1).tolist())
        # embeddings2 = torch.as_tensor(np.array([i['81'] for seq in seq_feat for i in seq[1:]]))
        # for i,(e1, e2) in enumerate(zip(embeddings, embeddings2)):
        #     if (e1 != e2).any():
        #         breakpoint()
        # print((embeddings == embeddings2).all())
        
        
        # 测试采样策略
        # sampler = BaseSampler(data_path='/home/yeep/project/alqq_generc/data/TencentGR_1k')
        
        # sampled_feat = sampler.sample(ids.reshape(-1).tolist())
        # for k,v in d1.items():
        #     if (sampled_feat[k] == v).all():
        #         print(k)
        #     else:
        #         breakpoint()
        
        
        
        
        
        
        