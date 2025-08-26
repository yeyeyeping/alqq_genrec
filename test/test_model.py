

from model.model import UserTower, ItemTower, BaselineModel
from dataset import MyDataset
from torch.utils.data import DataLoader
import const
import torch







def to_device(batch):
    return {k: v.cuda() for k, v in batch.items()}

if __name__ == "__main__": 
    dataset = MyDataset(const.data_path)
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
    # user_tower = UserTower().cuda()
    # for batch in dataloader:
    #     batch = to_device(batch)
    #     seq = batch['id']
    #     token_type = batch['token_type']
    #     print(user_tower(seq, token_type == 2, batch).shape)
        
    # item_tower = ItemTower().cuda()
    # for batch in dataloader:
    #     batch = to_device(batch)
    #     seq = batch['id']
    #     token_type = batch['token_type']
    #     batch['81'] = torch.randn((seq.shape[0], seq.shape[1], 32)).cuda()
    #     print(item_tower(seq, token_type == 1, batch).shape)
        
        
    model = BaselineModel().cuda()
    for batch in dataloader:
        batch = to_device(batch)
        seq = batch['id']
        token_type = batch['token_type']
        batch['81'] = torch.randn((seq.shape[0], seq.shape[1], 32)).cuda()
        print(model(seq,token_type, batch))