from dataset import MyDataset
from sampler import BaseSampler
def group_feat(batch):
    seq_id = batch['id']
    token_type = batch['token_type']
    feat = {k:v for k,v in batch.items() if k != 'id' and k != 'token_type'}
    return seq_id, token_type, feat
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import dataset2
    from mm_emb_loader import Memorymm81Embloader
    dataset = MyDataset(data_path='/home/yeep/project/alqq_generc/data/TencentGR_1k')
    dataloader = DataLoader(dataset, batch_size=77, )
    
    
    ds2 = dataset2.MyDataset(data_dir='/home/yeep/project/alqq_generc/data/TencentGR_1k')
    dataloader2 = DataLoader(ds2, batch_size=77, collate_fn=dataset2.MyDataset.collate_fn)
    emb_loader = Memorymm81Embloader(data_path='/home/yeep/project/alqq_generc/data/TencentGR_1k')
    for d1, d2 in zip(dataloader, dataloader2):
        seq_id, token_type,action_type, feat = d1
        breakpoint( )
        ids = d1["id"]
        seq1, pos = ids[:,:-1],ids[:,1:].clone()
        
        token_type1 = d1["token_type"]
        
        seq2, pos2, _, token_type2, _, _, seq_feat, _, _ = d2
        
        
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
        sampler = BaseSampler(data_path='/home/yeep/project/alqq_generc/data/TencentGR_1k')
        
        sampled_feat = sampler.reid2feat(ids[token_type1 == 1].reshape(-1).tolist())

        
        
        