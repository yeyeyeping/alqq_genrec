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
class NegDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.item_feat_dict = read_json(self.data_path / "item_feat_dict.json")
        self.item_num = list(range(1, len(self.item_feat_dict) + 1))
        
    def __len__(self):
        return 0x7FFFFFFF
    
    def __getitem__(self, index):
        neg_item_reid_list = []
        neg_item_feat_list = []
        for i in random.sample(self.item_num, 256):
            neg_item_reid_list.append(i)
            neg_item_feat_list.append(self.item_feat_dict[str(i)])
            
        return torch.as_tensor(neg_item_reid_list), MyDataset.collect_features(neg_item_feat_list, 
                                                                               include_user=False, 
                                                                               include_context=False)

class HotNegDataset(Dataset):
    def __init__(self, data_path, hot_exp_ratio=0.2, hot_click_ratio=0.05):
        self.data_path = Path(data_path)
        self.item_feat_dict = read_json(self.data_path / "item_feat_dict.json")
        self.item_num = list(range(1, len(self.item_feat_dict) + 1))
        item_expression_num,item_click_num = self._load_data_info()
        self.hot_expression = self.keep_hot_expression_item(item_expression_num)
        self.hot_click = self.keep_hot_click_item(item_click_num)
        self.hot_expression_ratio = hot_exp_ratio
        self.hot_click_ratio = hot_click_ratio
        print(f"hot expression item: {len(self.hot_expression)}, hot click item: {len(self.hot_click)}")
        
    def __len__(self):
        return 0x7FFFFFFF
    
    def keep_hot_expression_item(self,item_expression_num):
        hot_expression_item = []
        for k,v in item_expression_num.items():
            if v >= 5:
                hot_expression_item.append(k)
        return hot_expression_item
    
    def keep_hot_click_item(self,item_click_num):
        hot_click_item = []
        for k,v in item_click_num.items():
            if v >= 4:
                hot_click_item.append(k)
        return hot_click_item
        
    def _load_data_info(self):
        cache_path = Path(os.environ.get('USER_CACHE_PATH'))
        
        with open(cache_path/"data_info.pkl", "rb") as f:
            data_info = pickle.load(f)
        return data_info['item_expression_num'], data_info['item_click_num']
            
    def __getitem__(self, index):
        neg_item_reid_list = []
        neg_item_feat_list = []
        for i in range(256):
            sampled_id = 0
            p = random.random()
            if p < self.hot_click_ratio:
                sampled_id = random.choice(self.hot_click)
            elif p < self.hot_expression_ratio:
                sampled_id = random.choice(self.hot_expression)
            else:
                sampled_id = random.choice(self.item_num)
            
            neg_item_reid_list.append(sampled_id)
            neg_item_feat_list.append(self.item_feat_dict[str(sampled_id)])
            
            
        return torch.as_tensor(neg_item_reid_list), MyDataset.collect_features(neg_item_feat_list, 
                                                                               include_user=False, 
                                                                               include_context=False)

def collate_fn(batch):
    neg_item_reid_list, neg_item_feat_list = zip(*batch)
    reid = torch.cat(neg_item_reid_list)
    out_dict = {}
    for k in neg_item_feat_list[0].keys():
        feat = torch.cat([v[k] for v in neg_item_feat_list])
        out_dict[k] = feat

    return reid, out_dict

def sample_neg():
    dataset = None
    if const.sampling_strategy == 'random':
        dataset = NegDataset(const.data_path)
    elif const.sampling_strategy == 'hot':
        dataset = HotNegDataset(const.data_path)
    else:
        raise ValueError(f"Invalid sampling strategy: {const.sampling_strategy}")
    loader = DataLoader(dataset, 
                        batch_size=const.neg_sample_num // 256,
                        collate_fn=collate_fn,
                        num_workers=5,
                        prefetch_factor=2,
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
    