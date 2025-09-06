from .feature import user_feature, item_feature, context_feature
import os
from .model import ModelParam
from pathlib import Path

    

data_path = os.environ.get('TRAIN_DATA_PATH')
if data_path is None:
    print(f"Training data path is not set, switch to Test data path")
    data_path = os.environ.get('EVAL_DATA_PATH')
    if data_path is None:
        raise ValueError("Test data path is not set, please set the EVAL_DATA_PATH environment variable")
    
# 数据相关
mm_emb_dim = {
    "81": 32,
    "82": 1024,
    "83": 3584,
    "84": 4096,
    "85": 4096,
    "86": 3584,
}
max_seq_len = 101
# 训练相关
l2_alpha = 1e-7
device = "cuda"
batch_size = 128
num_workers = 8
num_epochs = 10
warmup_t = 2000
lr = 3e-3
seed = 3407
# sampling_strategy = "random"
# sampling_strategy = "hot" # hot
# hot_exp_ratio = 0.2
# hot_click_ratio = 0.05
sampling_strategy = "random"
uniform_sampling_ratio = 0.7
penalty_ratio = 0.5

# 采样池设置
num_sampled_once = 300
sampling_factor = 100
refresh_interval = 1000
neg_sample_num = 15000

temperature = 0.04
grad_norm = 1.0

# 推理相关
infer_batch_size = 512
# 模型参数
model_param = ModelParam(Path(data_path)/"indexer.pkl")
        




