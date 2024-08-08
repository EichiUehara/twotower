from dataset.movie_lens_base.UserDataset import UserDataset as UserDatasetBase
from module.DimCalculator import embedding_dim
class UserDataset(UserDatasetBase):
    def __init__(self, category):
        super().__init__(category)
        self.numerical_features = ['age', 'average_rating']
        self.categorical_features = ['occupation', 'zipcode', 'gender']
        self.text_features = []
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
    user_dataset = UserDataset()
    print(len(user_dataset))
    batch = [user_dataset[i] for i in user_dataset.dataframe.index[0:32]]
    print(batch)
    print(user_dataset.collate_fn(batch))