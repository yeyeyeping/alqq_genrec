from .feature import user_feature, item_feature, context_feature
import os
from .model import ModelParam
from pathlib import Path

    

data_path = os.environ.get('TRAIN_DATA_PATH')
data_path = Path(data_path)
user_cache_path = Path(os.environ.get('USER_CACHE_PATH'))

# 数据相关
mm_emb_dim = {
    "81": 32,
    "82": 1024,
    "83": 3584,
    "84": 4096,
    "85": 4096,
    "86": 3584,
}
max_seq_len = 102
# 训练相关
l2_alpha = 1e-7
device = "cuda"
batch_size = 128
num_workers = 8
num_epochs = 10
warmup_t = 2000
lr = 2e-3
seed = 3407
sampling_strategy = "hot" # hot
popularity_coef = 0.9
popularity_samples_ratio = 0.9
# hot_exp_ratio = 0.3
# hot_click_ratio = 0.05
neg_sample_num = 30000
temperature = 0.025
grad_norm = 1.0
# 推理相关
infer_batch_size = 512
# 模型参数
model_param = ModelParam(Path(data_path)/"indexer.pkl")
        




