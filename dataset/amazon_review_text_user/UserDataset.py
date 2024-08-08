import math
from dataset.amazon_review_base.UserDataset import UserDataset
from module.embedding_dim import embedding_dim
class UserDataset(UserDataset):
    def __init__(self, category):
        super().__init__(category)
        self.numerical_features = ['average_rating']
        self.categorical_features = []
        self.text_features = []
        self.history_features = ['purchased_item_ids']
        self.text_history_features = ['review_text_history']
        self.hyperparameters = {
            'id': {
                'num_classes': self.num_classes['id'], 
                'embedding_dim': embedding_dim(self.num_classes['id'])
            },
            'categorical_features': {
            },
            'text_features': {
            },
            'history_features': {
                'purchased_item_ids': {
                    'num_classes': self.num_classes['item_id'], 
                    'embedding_dim': embedding_dim(self.num_classes['item_id']),
                    'transformer_head': math.ceil(math.sqrt(math.sqrt(embedding_dim(self.num_classes['item_id'])))),
                }
            },
            'text_history_features': {
                'review_text_history': {
                }
            },
            'feedforward_network': {
            }
        }
        input_dim = 0
        input_dim += self.hyperparameters['id']['embedding_dim']
        for feature in self.numerical_features:
            input_dim += 1
        for feature in self.categorical_features:
            input_dim += self.hyperparameters['categorical_features'][feature]['embedding_dim']
        for feature in self.history_features:
            input_dim += self.hyperparameters['history_features'][feature]['embedding_dim']
        for feature in self.text_features:
            input_dim += 768
        for feature in self.text_history_features:
            input_dim += 768
        self.hyperparameters['feedforward_network']['input_dim'] = input_dim
if __name__ == '__main__':
    user_dataset = UserDataset('All_Beauty')
    print(len(user_dataset))
    batch = [user_dataset[i] for i in user_dataset.dataframe.index[0:32]]
    print(batch)
    print(user_dataset.collate_fn(batch))