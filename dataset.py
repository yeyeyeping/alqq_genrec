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
        for record_tuple in user_sequence:
            u, i, user_feat, item_feat, action_type, ts = record_tuple
            if u and user_feat:
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type, ts))
            if i and item_feat:
                ext_user_sequence.append((i, item_feat, 1, action_type, ts))
        return ext_user_sequence
    
    @classmethod
    def pad_seq(cls, seq, max_len, default_value):
        if len(seq) < max_len:
            if isinstance(seq, list):
                seq = [default_value,] * (max_len - len(seq)) + seq 
            elif isinstance(seq, torch.Tensor):
                default_value = torch.as_tensor(default_value)
                seq = torch.cat([default_value.repeat(max_len - len(seq)), seq])
            else:
                raise ValueError(f"Invalid default value type: {type(seq)}")
        elif len(seq) > max_len:
            print(f"seq length is greater than max_len, seq length: {len(seq)}, max_len: {max_len}")
            seq = seq[:max_len]
        else:
            pass
        return seq
    
    def norm_ts(self, ts: torch.Tensor) -> torch.Tensor:
        ts = ts.long()
        diffs = torch.diff(ts).abs()
        pos_diffs = diffs[diffs > 0]
        time_scale = pos_diffs.min() if pos_diffs.numel() > 0 else ts.new_tensor(1.0)
        norm = torch.round((ts - ts.min()) / time_scale).to(torch.long) + 1
        return norm
    
    def add_time_feat(self, ts_array):
        ts_tensor = torch.as_tensor(ts_array)
        time_gap = torch.diff(ts_tensor,prepend=ts_tensor[0][None])
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
        
        personal_diff = self.norm_ts(ts_tensor / 60 / 60)
        personal_diff = torch.diff(personal_diff) + 1
        personal_diff = torch.clamp(personal_diff, max=const.model_param.personal_time_span)
        
        
        out_time_feat = {
            "201": MyDataset.pad_seq(log_gap, const.max_seq_len + 1, 0),
            "202": MyDataset.pad_seq(weekdays, const.max_seq_len + 1, 0),
            "203": MyDataset.pad_seq(hours, const.max_seq_len + 1, 0),
            "204": MyDataset.pad_seq(months, const.max_seq_len + 1, 0),
            "205": MyDataset.pad_seq(days, const.max_seq_len  + 1, 0),
            "206": MyDataset.pad_seq(delta_scaled, const.max_seq_len + 1, 0),
            "207": MyDataset.pad_seq(personal_diff, const.max_seq_len + 1, 0)
            
        }
        return out_time_feat
        
    
    def __getitem__(self, index):
        user_seq = self._load_user_data(index)
        ext_user_seq = self.format_user_seq(user_seq)
        id_list = []
        token_type_list = []
        action_type_list = []
        feat_list = []
        ts_list = []
        front_click_item = []
        seq_list = []
        for i, feat, token_type, action_type, ts in ext_user_seq:
            id_list.append(i)
            token_type_list.append(token_type)
            action_type_list.append(action_type if action_type is not None else 0)
            feat_list.append(feat)
            
            seq_list.append(MyDataset.pad_seq(front_click_item[-const.context_feature.seq_len:].copy(), const.context_feature.seq_len, 0))
            if token_type == 1:
                ts_list.append(ts)
            if action_type == 1:
                front_click_item.append(i)
                
        time_feat = self.add_time_feat(ts_list)
        
        id_list = MyDataset.pad_seq(id_list, const.max_seq_len + 1, 0)
        token_type_list = MyDataset.pad_seq(token_type_list, const.max_seq_len + 1, 0)
        action_type_list = MyDataset.pad_seq(action_type_list, const.max_seq_len + 1, 0)
        feat_list = MyDataset.pad_seq(feat_list, const.max_seq_len + 1, {})
        seq_list = MyDataset.pad_seq(seq_list, const.max_seq_len + 1, [0] *const.context_feature.seq_len)
        return torch.as_tensor(id_list).int(), \
            torch.as_tensor(token_type_list).int(), \
                torch.as_tensor(action_type_list).int(), \
                    {
                        **MyDataset.collect_features(feat_list, include_context=False),
                        **time_feat,
                        "210": torch.as_tensor(seq_list, dtype=torch.int32)
                    }
                

    @classmethod
    def fill_feature(cls, feat,include_user=True, include_item=True, include_context=True):
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
    def collect_features(cls, feat_list, include_user=True, include_item=True, include_context=True):
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
    
    def norm_ts(self, ts: torch.Tensor) -> torch.Tensor:
        ts = ts.long()
        diffs = torch.diff(ts).abs()
        pos_diffs = diffs[diffs > 0]
        time_scale = pos_diffs.min() if pos_diffs.numel() > 0 else ts.new_tensor(1.0)
        norm = torch.round((ts - ts.min()) / time_scale).to(torch.long) + 1
        return norm
    
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
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type,ts))

            if i and item_feat:
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:
                    i = 0
                if item_feat:
                    item_feat = self._process_cold_start_feat(item_feat)
                ext_user_sequence.append((i, item_feat, 1,action_type, ts))
            
        return ext_user_sequence, user_id

    def add_time_feat(self, ts_array):
        ts_tensor = torch.as_tensor(ts_array)
        time_gap = torch.diff(ts_tensor,prepend=ts_tensor[0][None])
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
        
        personal_diff = self.norm_ts(ts_tensor)
        personal_diff = torch.diff(personal_diff) + 1
        personal_diff = torch.clamp(personal_diff, max=const.model_param.personal_time_span)
        
        
        out_time_feat = {
            "201": MyDataset.pad_seq(log_gap, const.max_seq_len, 0),
            "202": MyDataset.pad_seq(weekdays, const.max_seq_len , 0),
            "203": MyDataset.pad_seq(hours, const.max_seq_len , 0),
            "204": MyDataset.pad_seq(months, const.max_seq_len, 0),
            "205": MyDataset.pad_seq(days, const.max_seq_len , 0),
            "206": MyDataset.pad_seq(delta_scaled, const.max_seq_len, 0),
            "207": MyDataset.pad_seq(personal_diff, const.max_seq_len, 0)
            
        }
        return out_time_feat
    def __getitem__(self, uid):
        user_sequence = self._load_user_data(uid)
        ext_user_sequence, user_id = self.format_user_seq(user_sequence)
        
        id_list = []
        token_type_list = []
        feat_list = []
        ts_list = []
        
        front_click_item = []
        seq_list = []
        for i, feat, token_type,action_type,ts in ext_user_sequence:
            id_list.append(i)
            token_type_list.append(token_type)
            feat_list.append(feat)
            seq_list.append(MyDataset.pad_seq(front_click_item[-const.context_feature.seq_len:].copy(), const.context_feature.seq_len, 0))
            if token_type == 1:
                ts_list.append(ts)
                
            if action_type == 1:
                front_click_item.append(i)
            
        time_feat = self.add_time_feat(ts_list)
        
        
        
        id_list = MyDataset.pad_seq(id_list, const.max_seq_len, 0)
        token_type_list = MyDataset.pad_seq(token_type_list, const.max_seq_len, 0)
        feat_list = MyDataset.pad_seq(feat_list, const.max_seq_len, {})
        seq_list = MyDataset.pad_seq(seq_list, const.max_seq_len, [0] *const.context_feature.seq_len)
                
        return torch.as_tensor(id_list).int(), \
            torch.as_tensor(token_type_list).int(), \
                    {**MyDataset.collect_features(feat_list, include_context=False), \
                     **time_feat,
                     "210": torch.as_tensor(seq_list, dtype=torch.int32)},\
                    user_id


if __name__ == "__main__":
    dataset = MyDataset(data_path='/home/yeep/project/alqq_generc/data/TencentGR_1k')
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    for d in dataloader:
        print(d)
        breakpoint()
    


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
        
        
        
        
        
        
        