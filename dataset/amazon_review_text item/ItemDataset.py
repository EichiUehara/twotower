from dataset.amazon_review_base.ItemDataset import ItemDataset
from module.DimCalculator import embedding_dim
class ItemDataset(ItemDataset):
    def __init__(self, category):
        super().__init__(category)
        self.numerical_features = ['average_rating', 'rating_number']
        self.categorical_features = ['main_category', 'store']
        self.text_features = ['details']
        self.history_features = []
        self.text_history_features = []
        self.hyperparameters = {
            'categorical_features': {
                'main_category': {
                    'num_classes': self.num_classes['main_category'], 
                    'embedding_dim': embedding_dim(self.num_classes['main_category'])
                },
            },
            'text_features': {
            },
            'history_features': {
            },
            'text_history_features': {
            }
        }
if __name__ == '__main__':
    item_dataset = ItemDataset('All_Beauty')
    print(len(item_dataset))
    batch = [item_dataset[i] for i in item_dataset.dataframe.index[0:32]]
    print(batch)
    print(item_dataset.collate_fn(batch))