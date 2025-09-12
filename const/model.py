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
        self.embedding_table_size["403"] = 3
        self.embedding_table_size["402"] = 3
        self.embedding_table_size["123"] = 233
        
        self.embedding_dim = {
            "user_id":64,
            "item_id":64,
            # 物品特征
            "100": 16,      # 6
            "101": 16,     # 51
            "102": 32,     # 90709
            "111": 32,    # 4783154
            "112": 16,     # 30
            "114": 16,     # 20
            "115": 16,     # 691
            "116": 16,     # 18
            "117": 16,     # 497
            "118": 16,     # 1426
            "119": 24,     # 4191
            "120": 24,     # 3392
            "121": 32,    # 2135891
            "122": 32,     # 90919
            "123": 16,     # time

            # 多模态特征
            "81": 32,
            # "82": 128,
            # "83": 128,
            # "84": 128,
            # "85": 128,
            # "86": 128,
            # 用户特征
            "103": 16,     # 86
            "104": 16,      # 2
            "105": 16,      # 7
            "106": 16,     # 14
            "107": 16,     # 19
            "108": 16,      # 4
            "109": 16,      # 3
            "110": 16,       # 2
            
            "201": 32,# 时间特征
            "202": 16,# weekday
            "203": 16, # hour
            "204": 16, # month
            "205": 16, # day   
            "206": 16, # decay
            "402": 16, # action type
            "403": 16, # next action type
        }
        self.user_dnn_units = 128
        self.item_dnn_units = 128
        self.context_dnn_units = 128
        self.dropout = 0.2
        self.hidden_units = 256
        self.num_blocks = 8
        self.num_heads = 8
        self.norm_first = True
        self.relative_attention_num_buckets = 32
        self.relative_attention_bucket_dim = 16
        self.relative_attention_max_distance = 128
        self.num_experts = 10
    def read_feature_size(self, indexer_file):
        indexer = read_pickle(indexer_file)
        emb_table_size = {}
        emb_table_size['item_id'] = len(indexer['i'])
        emb_table_size['user_id'] = len(indexer['u'])

        for feat_id, mapping_dict in indexer['f'].items():
            emb_table_size[feat_id] = len(mapping_dict)
            
        return emb_table_size