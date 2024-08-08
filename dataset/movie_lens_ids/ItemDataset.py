from dataset.movie_lens_base.ItemDataset import ItemDataset as ItemDatasetBase
from module.DimCalculator import embedding_dim
class ItemDataset(ItemDatasetBase):
    def __init__(self, category):
        super().__init__(category)
        self.numerical_features = ['average_rating']
        self.categorical_features = ['genres', 'year']
        self.text_features = []
        self.history_features = []
        self.text_history_features = []
        self.hyperparameters = {
            'categorical_features': {
                'genres': {
                    'num_classes': self.num_classes['genres'], 
                    'embedding_dim': embedding_dim(self.num_classes['genres'])
                },
                'year': {
                    'num_classes': self.num_classes['year'], 
                    'embedding_dim': embedding_dim(self.num_classes['year'])
                }
            },
            'text_features': {
            },
            'history_features': {
            },
            'text_history_features': {
            }
        }
        input_dim = 0
        input_dim += self.hyperparameters['id']['embedding_dim']
        for feature in self.numerical_features:
            input_dim += 1
        for feature in self.hyperparameters['categorical_features']:
            input_dim += self.hyperparameters['categorical_features'][feature]['embedding_dim']
        for feature in self.history_features:
            input_dim += self.hyperparameters['categorical_features'][feature]['embedding_dim']
        for feature in self.text_features:
            input_dim += 768
        for feature in self.text_history_features:
            input_dim += 768
        self.feedforward_network = {
                'input_dim': input_dim,
                'hidden_dim': 512,
                'output_dim': 256
            }        

if __name__ == '__main__':
    item_dataset = ItemDataset()
    print(len(item_dataset))
    batch = [item_dataset[i] for i in item_dataset.dataframe.index[0:32]]
    print(batch)
    print(item_dataset.collate_fn(batch))