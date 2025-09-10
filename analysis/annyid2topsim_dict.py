from pathlib import Path
import os
import numpy as np
import pickle
import torch
from collections import defaultdict
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def read_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False


def load_embeddings_to_numpy(pkl_path: Path):
    data = read_pickle(pkl_path)
    # 推断维度
    first_vec = next(v for v in data.values() if isinstance(v, np.ndarray))
    dim = int(first_vec.shape[-1])
    num = len(data)

    ids = np.empty(num, dtype=np.int64)
    embs = np.empty((num, dim), dtype=np.float32)
    idx = 0
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            ids[idx] = int(k)
            embs[idx] = v.astype(np.float32, copy=False)
            idx += 1
    if idx != num:
        ids = ids[:idx]
        embs = embs[:idx]
    return ids, embs


def build_faiss_index(embs: np.ndarray, nlist: int, pq_m: int, use_gpu: bool):
    dim = embs.shape[1]
    # 归一化到单位范数，使内积等同于cosine相似度
    faiss.normalize_L2(embs)

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, 8)
    index.metric_type = faiss.METRIC_INNER_PRODUCT

    # 训练样本规模：nlist*200 或者 100万，取最大且不超过全集
    train_samples = min(embs.shape[0], max(1_000_000, nlist * 200))
    rng = np.random.default_rng(123)
    train_idx = rng.choice(embs.shape[0], size=train_samples, replace=False)
    index.train(embs[train_idx])

    if use_gpu and HAS_FAISS and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        # 使用 64 位索引以避免当原始 ID 超过 2^31-1 时的溢出
        co.indicesOptions = faiss.INDICES_64_BIT
        # 单卡优先，避免多卡复制导致显存翻倍
        index = faiss.index_cpu_to_gpu(res, 0, index, co)

    return index


def torch_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps))


def kmeans_train_torch(sample: torch.Tensor, nlist: int, iters: int = 20, seed: int = 123) -> torch.Tensor:
    torch.manual_seed(seed)
    device = sample.device
    S, D = sample.shape
    perm = torch.randperm(S, device=device)
    init_idx = perm[:nlist]
    centroids = sample[init_idx].clone()
    for _ in range(iters):
        sums = torch.zeros_like(centroids)
        counts = torch.zeros(nlist, device=device, dtype=torch.float32)
        bs = 200_000
        for s in range(0, S, bs):
            e = min(s + bs, S)
            x = sample[s:e]
            sim = x @ centroids.T
            labels = sim.argmax(dim=1)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
            sums.index_add_(0, labels, x)
        mask_empty = counts == 0
        if mask_empty.any():
            replace_idx = torch.randperm(S, device=device)[:mask_empty.sum()]
            centroids[mask_empty] = sample[replace_idx]
            counts = counts.masked_fill(mask_empty, 1.0)
        centroids = sums / counts.clamp_min(1e-6).unsqueeze(1)
        centroids = torch_normalize(centroids, dim=1)
    return centroids


def assign_primary_clusters(embs: torch.Tensor, centroids: torch.Tensor, bs: int = 100_000) -> torch.Tensor:
    N = embs.shape[0]
    out = torch.empty(N, dtype=torch.int32, device=embs.device)
    for s in range(0, N, bs):
        e = min(s + bs, N)
        sim = embs[s:e] @ centroids.T
        out[s:e] = sim.argmax(dim=1).to(torch.int32)
    return out


def topn_neighbor_clusters(centroids: torch.Tensor, nprobe: int) -> torch.Tensor:
    device = centroids.device
    nlist = centroids.shape[0]
    out = torch.empty((nlist, nprobe), dtype=torch.int32, device=device)
    step = 1024
    for s in range(0, nlist, step):
        e = min(s + step, nlist)
        sim = centroids[s:e] @ centroids.T
        _, idx = torch.topk(sim, k=nprobe, dim=1)
        out[s:e] = idx.to(torch.int32)
    return out


def torch_ivf_search(ids: np.ndarray, embs_np: np.ndarray, nlist: int, nprobe: int, k: int = 21,
                     kmeans_sample: int = 1_000_000, kmeans_iters: int = 20,
                     q_batch: int = 150_000, device: str = 'cuda'):
    x_all = torch.from_numpy(embs_np).to(device)
    x_all = torch_normalize(x_all, dim=1)

    rng = np.random.default_rng(123)
    samp = min(kmeans_sample, x_all.shape[0])
    samp_idx = torch.from_numpy(rng.choice(x_all.shape[0], size=samp, replace=False)).to(device)
    sample = x_all[samp_idx]
    centroids = kmeans_train_torch(sample, nlist=nlist, iters=kmeans_iters)

    primary = assign_primary_clusters(x_all, centroids, bs=100_000)
    nlist_int = int(nlist)
    lists = [[] for _ in range(nlist_int)]
    for i in range(primary.numel()):
        lists[int(primary[i].item())].append(i)
    lists = [torch.as_tensor(lst, dtype=torch.int64, device=device) if len(lst) > 0 else torch.empty(0, dtype=torch.int64, device=device) for lst in lists]

    neigh_clusters = topn_neighbor_clusters(centroids, nprobe=nprobe)

    n = x_all.shape[0]
    neighbors_out = np.empty((n, k - 1), dtype=np.int64)

    ids_tensor = torch.from_numpy(ids)
    for c in range(nlist_int):
        q_idx = lists[c]
        if q_idx.numel() == 0:
            continue
        cand_clusters = neigh_clusters[c].tolist()
        cand_idx_list = [lists[int(cc)].cpu() for cc in cand_clusters if lists[int(cc)].numel() > 0]
        if len(cand_idx_list) == 0:
            continue
        cand_idx = torch.unique(torch.cat(cand_idx_list))
        for s in range(0, q_idx.numel(), q_batch):
            e = min(s + q_batch, q_idx.numel())
            qb_idx = q_idx[s:e]
            q = x_all[qb_idx]
            cands = x_all[cand_idx.to(device)]
            sim = q @ cands.T
            _, idx_k = torch.topk(sim, k=k, dim=1)
            picked = cand_idx[idx_k.to('cpu')]
            picked_ids = ids_tensor[picked]
            self_ids = ids_tensor[qb_idx.to('cpu')].unsqueeze(1)
            mask = picked_ids != self_ids
            for r in range(picked_ids.shape[0]):
                row = picked_ids[r][mask[r]]
                row = row[:(k - 1)]
                if row.numel() < (k - 1):
                    pad = torch.full((k - 1 - row.numel(),), -1, dtype=row.dtype)
                    row = torch.cat([row, pad], dim=0)
                neighbors_out[qb_idx[s + r].item()] = row.numpy()

    return neighbors_out


def main():
    oov = []
    data_path = Path(os.environ.get('TRAIN_DATA_PATH'))
    cache_path = Path(os.environ.get('USER_CACHE_PATH'))
    emb_path = data_path / "creative_emb" / "emb_81_32.pkl"

    ids, embs = load_embeddings_to_numpy(emb_path)
    n, dim = embs.shape
    print(f"Loaded embeddings: n={n}, dim={dim}")

    backend = os.environ.get('ANN_BACKEND', 'faiss').lower()
    k = 21

    if backend == 'torch_ivf':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        nlist = int(min(16384, max(4096, 2 * int(np.sqrt(max(1, n))))))
        nprobe = 48
        neighbors_out = torch_ivf_search(ids, embs, nlist=nlist, nprobe=nprobe, k=k, device=device)
        all_ids = ids
    else:
        if not HAS_FAISS:
            raise ImportError("faiss is required for fast ANN search. Set ANN_BACKEND=torch_ivf to use PyTorch IVF.")
        nlist = int(min(65536, max(4096, 4 * int(np.sqrt(max(1, n))))))
        # 为更高召回选择尽可能大的可整除 pq_m（不超过32，且 >=4）
        target_m = min(32, dim)
        pq_m = None
        for m in range(target_m, 3, -1):
            if dim % m == 0:
                pq_m = m
                break
        if pq_m is None:
            pq_m = 4 if dim >= 4 else dim
        use_gpu = True

        index = build_faiss_index(embs, nlist=nlist, pq_m=pq_m, use_gpu=use_gpu)
        index.add_with_ids(embs, ids.astype(np.int64))
        if hasattr(index, 'nprobe'):
            index.nprobe = min(128, nlist)

        batch = 1_000_000
        neighbors_chunks = []
        id_batches = []
        for start in range(0, n, batch):
            end = min(start + batch, n)
            q = embs[start:end]
            q_ids = ids[start:end]
            D, I = index.search(q, k)
            self_first = (I[:, 0] == q_ids)
            neigh = np.empty((end - start, 20), dtype=np.int64)
            if np.any(self_first):
                neigh[self_first] = I[self_first, 1:21]
            if np.any(~self_first):
                rows = np.where(~self_first)[0]
                for r in rows:
                    row = I[r]
                    row = row[(row != -1) & (row != q_ids[r])]
                    if row.shape[0] < 20:
                        pad = np.full(20 - row.shape[0], -1, dtype=np.int64)
                        row = np.concatenate([row, pad], axis=0)
                    neigh[r] = row[:20]
            neighbors_chunks.append(neigh)
            id_batches.append(q_ids)
            if (start // batch) % 10 == 0:
                print(f"Processed {end}/{n}")

        neighbors_out = np.vstack(neighbors_chunks)
        all_ids = np.concatenate(id_batches)
    # 保存为 dict
    result = {int(all_ids[i]): [int(x) for x in neighbors_out[i] if x != -1] for i in range(all_ids.shape[0])}
    index_file = data_path / "indexer.pkl"
    indexer = read_pickle(index_file)
    result_dict = defaultdict(list)
    for k, v in result.items():
        if k in indexer['i']:
            for x in v:
                if x in indexer['i']:
                    result_dict[indexer['i'][k]].append(indexer['i'][x])
                else:
                    oov.append(k)
        else:
            oov.append(k)
    
    out_path = cache_path / "annoyid2top20sim_dict.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"save to {out_path}")
    for o in oov:
        print(o)

    for i, (k, v) in enumerate(result_dict.items()):
        print(k, v)
        if i > 100:
            break

    print(f"oov: {len(oov)}")
    


if __name__ == "__main__":
    main()

