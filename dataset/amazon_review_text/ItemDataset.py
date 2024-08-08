from dataset.amazon_review_base.ItemDataset import ItemDataset
from module.embedding_dim import embedding_dim
class ItemDataset(ItemDataset):
    def __init__(self, category):
        super().__init__(category)
        self.numerical_features = ['average_rating', 'rating_number']
        self.categorical_features = ['main_category', 'store']
        self.text_features = ['details']
        self.history_features = []
        self.text_history_features = []
        self.hyperparameters = {
            'id': {
                'num_classes': self.num_classes['id'], 
                'embedding_dim': embedding_dim(self.num_classes['id'])
            },
            'categorical_features': {
                'main_category': {
                    'num_classes': self.num_classes['main_category'], 
                    'embedding_dim': embedding_dim(self.num_classes['main_category'])
                },
                'store': {
                    'num_classes': self.num_classes['store'], 
                    'embedding_dim': embedding_dim(self.num_classes['store'])
                }
            },
            'text_features': {
                'details': {
                }
            },
            'history_features': {
            },
            'text_history_features': {
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
    item_dataset = ItemDataset('All_Beauty')
    print(len(item_dataset))
    batch = [item_dataset[i] for i in item_dataset.dataframe.index[0:32]]
    print(batch)
    print(item_dataset.collate_fn(batch))