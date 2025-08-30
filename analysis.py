import numpy as np  
from pathlib import Path
import os
import jsonlines
from collections import defaultdict
import pickle
data_path = Path(os.environ.get('TRAIN_DATA_PATH'))
# cache_path = Path(os.environ.get('USER_CACHE_PATH'))


# max_ts = float('-inf')
# min_ts = float('inf')
# for interactive_records in jsonlines.open(data_path/"seq.jsonl"):
    
#     for r in interactive_records:
#         u, i, user_feat, item_feat, action_type, ts = r
        
#         min_ts = min(min_ts, ts)
#         max_ts = max(max_ts, ts)
# print(f"min_ts: {min_ts}")
# print(f"max_ts: {max_ts}")
MIN_TS = 1728921670
import torch

def norm_ts(ts: torch.Tensor) -> torch.Tensor:
    diffs = torch.diff(ts)
    pos_diffs = diffs[diffs > 0]
    time_scale = pos_diffs.min() if pos_diffs.numel() > 0 else ts.new_tensor(1.0)
    norm = torch.round((ts - ts.min() - MIN_TS) / time_scale).to(torch.long) + 1
    return norm
max_norm_time = float('-inf')
min_norm_time = float('inf')
for interactive_records in jsonlines.open(data_path/"seq.jsonl"):
    time_list = []
    for r in interactive_records:
        *_, ts = r
        time_list.append(ts)
        
    time_list = norm_ts(torch.as_tensor(time_list))
    max_norm_time = max(max_norm_time, time_list.max())
    min_norm_time = min(min_norm_time, time_list.min())
    print(time_list)
print(f"max_norm_time: {max_norm_time}")
        