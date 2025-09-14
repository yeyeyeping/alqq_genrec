from utils import read_json
from dataset import MyDataset
import torch
from pathlib import Path
import random
import multiprocessing as mp
from torch.utils.data import Dataset,DataLoader
import const
import os
import pickle
from collections import defaultdict

MEAN_TIME = 48.32138517426633
MAX_TIME = 231.31589120370373
class NegDataset(Dataset):
    def __init__(self, data_path, time_dict):
        self.data_path = Path(data_path)
        self.item_feat_dict = read_json(self.data_path / "item_feat_dict.json")
        # 连续 item id: [1, N]
        self._num_items = len(self.item_feat_dict)
        # 使用张量缓冲区实现接近“无放回”的均匀采样，避免 Python random 开销
        self._uni_buffer_size = max(5_000_000, 256 * 1024)
        self._uni_buffer = None
        self._uni_ptr = 0
        self.time_dict = time_dict
        
    def __len__(self):
        return 0x7FFFFFFF
    
    def __getitem__(self, index):
        if self._uni_buffer is None:
            self._refill_uniform_buffer()
        neg_item_reid_list = []
        neg_item_feat_list = []
        # 从缓冲区取样，近似无放回（去重不足部分再补齐）
        sampled_id = self._draw_uniform_ids(256)
        for i in sampled_id.tolist():
            neg_item_reid_list.append(i)
            feat = self.item_feat_dict[str(i)]
            feat['123'] = self.time_dict[i] if i in self.time_dict else MEAN_TIME
            feat['123'] = int(feat['123']) + 1
            neg_item_feat_list.append(feat)
            
        return torch.as_tensor(neg_item_reid_list), MyDataset.collect_features(neg_item_feat_list, 
                                                                               include_item=True, 
                                                                               include_context=False, 
                                                                               include_user=False)
    def _refill_uniform_buffer(self):
        # 大批量有放回均匀采样，后续在 batch 级别去重
        self._uni_buffer = torch.randint(1, self._num_items + 1, (self._uni_buffer_size,), dtype=torch.long)
        self._uni_ptr = 0

    def _draw_uniform_ids(self, k: int) -> torch.Tensor:
        if k <= 0:
            return torch.empty(0, dtype=torch.long)
        # 若缓冲区不够则补充
        if self._uni_ptr + k > self._uni_buffer.numel():
            self._refill_uniform_buffer()
        out = self._uni_buffer[self._uni_ptr:self._uni_ptr + k]
        self._uni_ptr += k
        # 尽量无放回：去重，不足部分从缓冲区补齐一次
        if out.numel() > 1:
            uniq = torch.unique(out, sorted=False)
            if uniq.numel() < k:
                need = k - uniq.numel()
                if self._uni_ptr + need > self._uni_buffer.numel():
                    self._refill_uniform_buffer()
                supplement = self._uni_buffer[self._uni_ptr:self._uni_ptr + need]
                self._uni_ptr += need
                out = torch.cat([uniq, supplement], dim=0)
            else:
                out = uniq[:k]
        return out

class HotNegDataset(Dataset):
    def __init__(self, data_path, time_dict):
        self.data_path = Path(data_path)
        self.item_feat_dict = read_json(self.data_path / "item_feat_dict.json")
        self.hot_exp_items_list, self.cold_items_list = self._load_data_info()
        self.item_id2_time_dict = time_dict
        print(f"hot expression item: {len(self.hot_exp_items_list)}, cold item: {len(self.cold_items_list)}")
        # buffered uniform samplers for hot/cold pools
        self._buf_size = max(5_000_000, 256 * 1024)
        self._hot_buf = None
        self._hot_ptr = 0
        self._cold_buf = None
        self._cold_ptr = 0
        
    def __len__(self):
        return 0x7FFFFFFF
    
    
    def _load_data_info(self):
        cache_path = Path(os.environ.get('USER_CACHE_PATH'))
        
        with open(cache_path/"data_info.pkl", "rb") as f:
            data_info = pickle.load(f)
        
        hot_exp_items_list = []
        cold_items_list = []
        for k,v in data_info['item_expression_num'].items():
            if v > 10:
                hot_exp_items_list.append(k)
            else:
                cold_items_list.append(k)
                 
        return hot_exp_items_list, cold_items_list
            
    def __getitem__(self, index):
        if self._hot_buf is None:
            self._refill_hot()
        if self._cold_buf is None:
            self._refill_cold()
        num_hot_exp = int(256 * const.hot_exp_ratio)
        num_cold = 256 - num_hot_exp
        hot_ids = self._draw_from_pool('hot', num_hot_exp)
        cold_ids = self._draw_from_pool('cold', num_cold)
        item_ids = torch.cat([hot_ids, cold_ids], dim=0)
        neg_item_reid_list = item_ids
        neg_item_feat_list = []
        for i in item_ids.tolist():
            feat = self.item_feat_dict[str(i)]
            feat['123'] = self.item_id2_time_dict[i] if i in self.item_id2_time_dict else MEAN_TIME
            feat['123'] = int(feat['123']) + 1
            neg_item_feat_list.append(feat)
          
        return torch.as_tensor(neg_item_reid_list), MyDataset.collect_features(neg_item_feat_list, 
                                                                               include_user=False, 
                                                                               include_context=False)
    def _refill_hot(self):
        pool = torch.as_tensor(self.hot_exp_items_list, dtype=torch.long)
        idx = torch.randint(0, pool.numel(), (self._buf_size,), dtype=torch.long)
        self._hot_buf = pool[idx]
        self._hot_ptr = 0

    def _refill_cold(self):
        pool = torch.as_tensor(self.cold_items_list, dtype=torch.long)
        idx = torch.randint(0, pool.numel(), (self._buf_size,), dtype=torch.long)
        self._cold_buf = pool[idx]
        self._cold_ptr = 0

    def _draw_from_pool(self, which: str, k: int) -> torch.Tensor:
        if k <= 0:
            return torch.empty(0, dtype=torch.long)
        if which == 'hot':
            if self._hot_ptr + k > self._hot_buf.numel():
                self._refill_hot()
            out = self._hot_buf[self._hot_ptr:self._hot_ptr + k]
            self._hot_ptr += k
        else:
            if self._cold_ptr + k > self._cold_buf.numel():
                self._refill_cold()
            out = self._cold_buf[self._cold_ptr:self._cold_ptr + k]
            self._cold_ptr += k
        # dedup within draw; top up once if needed
        if out.numel() > 1:
            uniq = torch.unique(out, sorted=False)
            if uniq.numel() < k:
                need = k - uniq.numel()
                if which == 'hot':
                    if self._hot_ptr + need > self._hot_buf.numel():
                        self._refill_hot()
                    supplement = self._hot_buf[self._hot_ptr:self._hot_ptr + need]
                    self._hot_ptr += need
                else:
                    if self._cold_ptr + need > self._cold_buf.numel():
                        self._refill_cold()
                    supplement = self._cold_buf[self._cold_ptr:self._cold_ptr + need]
                    self._cold_ptr += need
                out = torch.cat([uniq, supplement], dim=0)
            else:
                out = uniq[:k]
        return out
    

def collate_fn(batch):
    neg_item_reid_list, neg_item_feat_list = zip(*batch)
    reid = torch.cat(neg_item_reid_list)
    out_dict = {}
    for k in neg_item_feat_list[0].keys():
        feat = torch.cat([v[k] for v in neg_item_feat_list])
        out_dict[k] = feat

    return reid, out_dict

def sample_neg(time_dict):
    dataset = None
    if const.sampling_strategy == 'random':
        dataset = NegDataset(const.data_path, time_dict)
    elif const.sampling_strategy == 'hot':
        dataset = HotNegDataset(const.data_path, time_dict)
    else:
        raise ValueError(f"Invalid sampling strategy: {const.sampling_strategy}")
    loader = DataLoader(dataset, 
                        batch_size=const.neg_sample_num // 256,
                        collate_fn=collate_fn,
                        pin_memory=True,
                        # prefetch_factor=4,
                        num_workers=4,
                        )
    
    return loader
            
# class BaseSampler:
#     def __init__(self, data_path):
#         self.data_path = Path(data_path)
#         self.item_feat_dict = read_json(self.data_path / "item_feat_dict.json")
        
#     def reid2feat(self, item_reid_list):
#         item_reid_list = sorted(item_reid_list)
#         item_feat_list = [MyDataset.fill_feature(self.item_feat_dict[str(i)]) for i in item_reid_list]
#         item_feat_list = MyDataset.collect_features(item_feat_list)
#         return torch.as_tensor(item_reid_list),item_feat_list
        
    
# class RandomSampler(BaseSampler):
#     def __init__(self, data_path,):
#         super().__init__(data_path)
#         self.item_ids = list(range(1, len(self.item_feat_dict) + 1))

#     def sample(self, num_samples, exclude_item_reid_list):
#         # 抽样num_samples个item，排除exclude_item_reid_list中的item
#         exclude_item_reid_list = set(exclude_item_reid_list)
#         sampled_item_reid_list = set()

#         while num_samples > 0:
#             sampled_reid = random.sample(self.item_ids, num_samples)
#             sampled_item_reid_list.update((i for i in sampled_reid if i not in exclude_item_reid_list))
#             num_samples = num_samples - len(sampled_item_reid_list)
#         return self.reid2feat(list(sampled_item_reid_list))
        

# if __name__ == "__main__":
#     # sampler = RandomSampler(data_path='/home/yeep/project/alqq_generc/data/TencentGR_1k')
#     # for i in range(100):    
#     #     sampled = random.randint(0, 400)
#     #     sampled_feat = sampler.sample(sampled, random.sample(range(1, sampler.item_num + 1), sampled//10))
#     #     assert sampled == len(sampled_feat['id']),f"sampled: {sampled}, len(sampled_feat['id']): {len(sampled_feat['id'])}"
#     #     print(f"passed {sampled}")
#     loader = sample_neg()
#     for reid, feat in loader:
#         breakpoint()
    