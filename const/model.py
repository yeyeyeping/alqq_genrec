from utils import read_pickle
class ModelParam:
    def __init__(self,indexer_file):
        self.embedding_table_size = self.read_feature_size(indexer_file)
        self.max_decay = 20
        self.max_diff = 20
        self.embedding_table_size["201"] = self.max_diff
        self.embedding_table_size["202"] = 8
        self.embedding_table_size["203"] = 25
        self.embedding_table_size["204"] = 13
        self.embedding_table_size["205"] = 32
        self.embedding_table_size["206"] = self.max_decay
        
        self.embedding_dim = {
            "user_id":256,
            "item_id":256,
            # 物品特征
            "100": 4,      # 6
            "101": 8,     # 51
            "102": 32,     # 90709
            "111": 64,    # 4783154
            "112": 6,     # 30
            "114": 6,     # 20
            "115": 16,     # 691
            "116": 6,     # 18
            "117": 16,     # 497
            "118": 16,     # 1426
            "119": 16,     # 4191
            "120": 16,     # 3392
            "121": 64,    # 2135891
            "122": 32,     # 90919

            # 多模态特征
            "81": 64,
            "82": 128,
            "83": 128,
            "84": 128,
            "85": 128,
            "86": 128,
            # 用户特征
            "103": 12,     # 86
            "104": 8,      # 2
            "105": 8,      # 7
            "106": 8,     # 14
            "107": 8,     # 19
            "108": 8,      # 4
            "109": 8,      # 3
            "110": 8,       # 2
            
            "201": 32,# 时间特征
            "202": 32,# weekday
            "203": 32, # hour
            "204": 32, # month
            "205": 32, # day   
            "206": 32, # decay
        }
        
        self.user_dnn_units = 128
        self.item_dnn_units = 128
        self.context_dnn_units = 128
        self.dropout = 0.2
        self.hidden_units = 256
        self.num_blocks = 8
        self.num_heads = 8
        self.norm_first = True
    def read_feature_size(self, indexer_file):
        indexer = read_pickle(indexer_file)
        emb_table_size = {}
        emb_table_size['item_id'] = len(indexer['i'])
        emb_table_size['user_id'] = len(indexer['u'])

        for feat_id, mapping_dict in indexer['f'].items():
            emb_table_size[feat_id] = len(mapping_dict)
            
        return emb_table_size