from dataset.amazon_review_base.UserDataset import UserDataset
from module.DimCalculator import embedding_dim
class UserDataset(UserDataset):
    def __init__(self, category):
        super().__init__(category)
        self.numerical_features = ['average_rating']
        self.categorical_features = []
        self.text_features = []
        self.history_features = ['purchased_item_ids']
        self.text_history_features = []
        self.hyperparameters = {
            'categorical_features': {
            },
            'text_features': {
            },
            'history_features': {
                'purchased_item_ids': {
                    'num_classes': self.num_classes['item_id'], 
                    'embedding_dim': embedding_dim(self.num_classes['item_id']),
                    'max_history_length': 10
                }
            },
            'text_history_features': {
            }
        }
if __name__ == '__main__':
    user_dataset = UserDataset('All_Beauty')
    print(len(user_dataset))
    batch = [user_dataset[i] for i in user_dataset.dataframe.index[0:32]]
    print(batch)
    print(user_dataset.collate_fn(batch))