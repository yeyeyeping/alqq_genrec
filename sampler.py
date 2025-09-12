from utils import read_json, read_pickle
from dataset import MyDataset
import torch
from pathlib import Path
import random
import multiprocessing as mp
from torch.utils.data import Dataset,DataLoader
import const
import os
import pickle

MEAN_TIME = 48.32138517426633
MAX_TIME = 231.31589120370373
from collections import defaultdict
class NegDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.item_feat_dict = read_json(self.data_path / "item_feat_dict.json")
        # 连续 item id: [1, N]
        self._num_items = len(self.item_feat_dict)
        # 使用张量缓冲区实现接近“无放回”的均匀采样，避免 Python random 开销
        self._uni_buffer_size = max(2_000_000, 256 * 1024)
        self._uni_buffer = None
        self._uni_ptr = 0
        
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
            neg_item_feat_list.append(self.item_feat_dict[str(i)])
            
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
MEAN_CLIK_RATE = 0.09727
TAU = 10

class HotNegDataset(Dataset):
    def __init__(self, data_path, time_dict):
        self.data_path = Path(data_path)
        self.item_feat_dict = read_json(self.data_path / "item_feat_dict.json")
        self.item_ids = list(range(1, len(self.item_feat_dict) + 1))
        self.popularity = torch.as_tensor(self.calc_poplurity(), dtype=torch.float32)
        # 预生成热门采样缓冲区，避免每次都在 500w 权重上采样
        self._pop_buffer_size = max(500_000, 256 * 1024)
        self._pop_buffer = None
        self._pop_ptr = 0
        self.item_id2_time_dict = time_dict
    
    def calc_poplurity(self, ):
        item_expression_num,item_click_num = self._load_data_info()
        popularity = []
        for k in range(1, len(self.item_feat_dict) + 1):
            # 计算流行度, 并采用贝叶斯平滑
            exp = item_expression_num[str(k)]
            click = item_click_num[str(k)]
            p = (click + MEAN_CLIK_RATE * TAU) / (exp + TAU) 
            popularity.append(p ** const.popularity_coef)
        return popularity
                        
    def __len__(self):
        return 0x7FFFFFFF
        
    def _load_data_info(self):
        cache_path = Path(os.environ.get('USER_CACHE_PATH'))
        
        with open(cache_path/"data_info.pkl", "rb") as f:
            data_info = pickle.load(f)
        return data_info['item_expression_num'], data_info['item_click_num']
            
    def __getitem__(self, index):
        if self._pop_buffer is None:
            self._refill_pop_buffer()
        neg_item_reid_list = []
        neg_item_feat_list = []
        num_sampled_popularity = int(256 * const.popularity_samples_ratio)
        num_sampled_random = 256 - num_sampled_popularity
        # 从缓冲区取样，近似无放回（去重不足部分再补齐）
        sampled_id = self._draw_pop_ids(num_sampled_popularity)
        sampled_id = sampled_id.tolist() + random.sample(self.item_ids, num_sampled_random)
        
        for i in sampled_id:            
            neg_item_reid_list.append(i)

            
            feat = self.item_feat_dict[str(i)]            
            feat['123'] = self.item_id2_time_dict[i] if i in self.item_id2_time_dict else MEAN_TIME
            feat['123'] = int(feat['123']) + 1
            
                        
            neg_item_feat_list.append(feat)
            
        return torch.as_tensor(neg_item_reid_list), MyDataset.collect_features(neg_item_feat_list, 
                                                                               include_user=False, 
                                                                               include_context=False)

    def _refill_pop_buffer(self):
        # 使用一次性大规模有放回采样来填充缓冲区（C++实现，速度快），索引从1开始
        self._pop_buffer = torch.multinomial(self.popularity, self._pop_buffer_size, replacement=True) + 1
        self._pop_ptr = 0

    def _draw_pop_ids(self, k: int) -> torch.Tensor:
        if k <= 0:
            return torch.empty(0, dtype=torch.long)
        # 若缓冲区不够则补充
        if self._pop_ptr + k > self._pop_buffer.numel():
            self._refill_pop_buffer()
        out = self._pop_buffer[self._pop_ptr:self._pop_ptr + k]
        self._pop_ptr += k
        # 尽量无放回：去重，不足部分从缓冲区补齐一次
        if out.numel() > 1:
            uniq = torch.unique(out, sorted=False)
            if uniq.numel() < k:
                need = k - uniq.numel()
                if self._pop_ptr + need > self._pop_buffer.numel():
                    self._refill_pop_buffer()
                supplement = self._pop_buffer[self._pop_ptr:self._pop_ptr + need]
                self._pop_ptr += need
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
        # dataset = NegDataset(const.data_path)
        pass
    elif const.sampling_strategy == 'hot':
        dataset = HotNegDataset(const.data_path, time_dict)
    elif const.sampling_strategy == 'hot_expression':
        # dataset = HotExpressionNegDataset(const.data_path)
        pass
    else:
        raise ValueError(f"Invalid sampling strategy: {const.sampling_strategy}")
    loader = DataLoader(dataset, 
                        batch_size=const.neg_sample_num // 256,
                        collate_fn=collate_fn,
                        num_workers=4
                        )
    
    return loader
            
class HotExpressionNegDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.item_feat_dict = read_json(self.data_path / "item_feat_dict.json")
        self.item_num = list(range(1, len(self.item_feat_dict) + 1))
        item_expression_num = self._load_data_info()
        self.hot_expression, self.cold_expression = self.keep_hot_expression_item(item_expression_num)
        # 保证 id 为整型，便于张量化与后续拼接
        self.hot_expression = [int(i) for i in self.hot_expression]
        self.cold_expression = [int(i) for i in self.cold_expression]
        # 预生成缓冲区以优化 Python 循环开销
        self._buf_size = max(2_000_000, 256 * 1024)
        self._hot_ids_tensor = torch.as_tensor(self.hot_expression, dtype=torch.long) if len(self.hot_expression) > 0 else torch.empty(0, dtype=torch.long)
        self._cold_ids_tensor = torch.as_tensor(self.cold_expression, dtype=torch.long) if len(self.cold_expression) > 0 else torch.empty(0, dtype=torch.long)
        self._hot_buffer = None
        self._cold_buffer = None
        self._hot_ptr = 0
        self._cold_ptr = 0
        print(f"hot expression item: {len(self.hot_expression)}")
        
    def __len__(self):
        return 0x7FFFFFFF
    
    def keep_hot_expression_item(self,item_expression_num):
        hot_expression_item = []
        cold_expression_item = []
        for k,v in item_expression_num.items():
            if v >= 10:
                hot_expression_item.append(k)
            else:
                cold_expression_item.append(k)
        return hot_expression_item, cold_expression_item
    
    def _load_data_info(self):
        cache_path = Path(os.environ.get('USER_CACHE_PATH'))
        
        with open(cache_path/"data_info.pkl", "rb") as f:
            data_info = pickle.load(f)
        return data_info['item_expression_num']
    
    def _refill_hot_buffer(self):
        if self._hot_ids_tensor.numel() == 0:
            self._hot_buffer = torch.empty(0, dtype=torch.long)
        else:
            idx = torch.randint(0, self._hot_ids_tensor.numel(), (self._buf_size,), dtype=torch.long)
            self._hot_buffer = self._hot_ids_tensor[idx]
        self._hot_ptr = 0
    
    def _refill_cold_buffer(self):
        if self._cold_ids_tensor.numel() == 0:
            self._cold_buffer = torch.empty(0, dtype=torch.long)
        else:
            idx = torch.randint(0, self._cold_ids_tensor.numel(), (self._buf_size,), dtype=torch.long)
            self._cold_buffer = self._cold_ids_tensor[idx]
        self._cold_ptr = 0
    
    def _draw_hot_ids(self, k: int) -> torch.Tensor:
        if k <= 0:
            return torch.empty(0, dtype=torch.long)
        if self._hot_buffer is None or self._hot_ptr + k > self._hot_buffer.numel():
            self._refill_hot_buffer()
        out = self._hot_buffer[self._hot_ptr:self._hot_ptr + k]
        self._hot_ptr += k
        if out.numel() > 1:
            uniq = torch.unique(out, sorted=False)
            if uniq.numel() < k:
                need = k - uniq.numel()
                if self._hot_ptr + need > (0 if self._hot_buffer is None else self._hot_buffer.numel()):
                    self._refill_hot_buffer()
                supplement = self._hot_buffer[self._hot_ptr:self._hot_ptr + need]
                self._hot_ptr += need
                out = torch.cat([uniq, supplement], dim=0)
            else:
                out = uniq[:k]
        return out
    
    def _draw_cold_ids(self, k: int) -> torch.Tensor:
        if k <= 0:
            return torch.empty(0, dtype=torch.long)
        if self._cold_buffer is None or self._cold_ptr + k > self._cold_buffer.numel():
            self._refill_cold_buffer()
        out = self._cold_buffer[self._cold_ptr:self._cold_ptr + k]
        self._cold_ptr += k
        if out.numel() > 1:
            uniq = torch.unique(out, sorted=False)
            if uniq.numel() < k:
                need = k - uniq.numel()
                if self._cold_ptr + need > (0 if self._cold_buffer is None else self._cold_buffer.numel()):
                    self._refill_cold_buffer()
                supplement = self._cold_buffer[self._cold_ptr:self._cold_ptr + need]
                self._cold_ptr += need
                out = torch.cat([uniq, supplement], dim=0)
            else:
                out = uniq[:k]
        return out
            
    def __getitem__(self, index):
        # 每次返回 256 个负样本，30% 来自曝光≥10 的热门集合
        k = 256
        num_hot = int(k * const.hot_exp_ratio)
        num_cold = k - num_hot
        hot_ids = self._draw_hot_ids(num_hot)
        cold_ids = self._draw_cold_ids(num_cold)
        sampled_ids = torch.cat([hot_ids, cold_ids], dim=0)
        neg_item_reid_list = sampled_ids.tolist()
        neg_item_feat_list = [self.item_feat_dict[str(i)] for i in neg_item_reid_list]
        return torch.as_tensor(neg_item_reid_list), MyDataset.collect_features(neg_item_feat_list, 
                                                                               include_user=False, 
                                                                               include_context=False)
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
    