from pathlib import Path
import os
import numpy as np
import pickle
import torch
import gc
def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

data_path = Path(os.environ.get('TRAIN_DATA_PATH'))
cache_path = Path(os.environ.get('USER_CACHE_PATH'))

indexer_file = read_pickle(data_path / "indexer.pkl")
emb = data_path/"creative_emb"/"emb_81_32.pkl"

id_list = []
emb_list = []
for k, v in read_pickle(emb).items():
    if isinstance(v, np.ndarray):
        id_list.append(k)
        emb_list.append(torch.as_tensor(v, dtype=torch.float32,device=torch.device('cuda')))
id_tensors = torch.as_tensor(id_list,device=torch.device('cuda'),dtype=torch.int64)
emb_tensors = torch.stack(emb_list)

# 分块参数 - 用于控制内存使用，避免一次性加载所有embeddings
src_chunk_size = 20000  # 源embeddings的chunk大小，每次处理1000个源向量
emb_chunk_size = 60000  # 目标embeddings的chunk大小，每次计算相似度时目标向量分块大小

top21_list = []
total_items = len(id_list)
print("find top 20 sim item")
# 对源embeddings分块处理
for i in range(0, total_items, src_chunk_size):
    end_i = min(i + src_chunk_size, total_items)
    emb_src = emb_tensors[i:end_i]
    a_id = id_tensors[i:end_i]

    # 初始化相似度矩阵和索引矩阵
    all_similarities = []
    all_indices = []

    # 对目标embeddings分块计算相似度
    for j in range(0, total_items, emb_chunk_size):
        end_j = min(j + emb_chunk_size, total_items)
        emb_target = emb_tensors[j:end_j]
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        # 计算当前chunk的相似度
            sim_chunk = emb_src @ emb_target.T
        all_similarities.append(sim_chunk)

        # 创建对应的索引（全局索引）
        indices_chunk = torch.arange(j, end_j, device=sim_chunk.device).unsqueeze(0).expand(sim_chunk.shape[0], -1)
        all_indices.append(indices_chunk)

    # 拼接所有相似度chunks
    full_sim_mat = torch.cat(all_similarities, dim=1)
    full_indices_mat = torch.cat(all_indices, dim=1)

    # 找到top21（包含自己）
    _, topk_indices = torch.topk(full_sim_mat, k=21, dim=1)
    top21_global_indices = full_indices_mat.gather(1, topk_indices)
    top21 = id_tensors[top21_global_indices].cpu().tolist()
    del sim_chunk, indices_chunk, full_sim_mat, full_indices_mat, top21_global_indices, topk_indices
    gc.collect()
    top21_list.append(top21)
print("remove self similarity")

# 移除自相似项
for batch_idx in range(len(top21_list)):
    start_i = batch_idx * src_chunk_size
    end_i = min((batch_idx + 1) * src_chunk_size, total_items)
    a_id_batch = id_list[start_i:end_i]

    for idx, id_ in enumerate(a_id_batch):
        if id_ in top21_list[batch_idx][idx]:
            top21_list[batch_idx][idx].remove(id_)


top21 = []
for i in top21_list:
    top21.extend(i)

# annoy id to reid
for idx in range(len(top21)):
    top21[idx] = [indexer_file['i'][id_] for id_ in top21[idx]]
    id_list[idx] = indexer_file['i'][id_list[idx]]
            
annoyied_top21 = dict(zip(id_list, top21))       
with open(cache_path/"annoyid2top20sim_dict.pkl", "wb") as f:
    pickle.dump(annoyied_top21, f)
print(f"save to {cache_path/'annoyid2top20sim_dict.pkl'}")

