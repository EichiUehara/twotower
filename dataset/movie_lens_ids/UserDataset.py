from dataset.movie_lens_base.UserDataset import UserDataset as UserDatasetBase
from module.embedding_dim import embedding_dim
class UserDataset(UserDatasetBase):
    def __init__(self):
        super().__init__()
        self.numerical_features = ['age', 'average_rating']
        self.categorical_features = ['occupation', 'zipcode', 'gender']
        self.text_features = []
        self.history_features = []
        self.text_history_features = []
        self.hyperparameters = {
            'id': {
                'num_classes': self.num_classes['id'], 
                'embedding_dim': embedding_dim(self.num_classes['id'])
            },
            'categorical_features': {
                'occupation': {
                    'num_classes': self.num_classes['occupation'], 
                    'embedding_dim': embedding_dim(self.num_classes['occupation'])
                },
                'zipcode': {
                    'num_classes': self.num_classes['zipcode'], 
                    'embedding_dim': embedding_dim(self.num_classes['zipcode'])
                },
                'gender': {
                    'num_classes': self.num_classes['gender'],
                    'embedding_dim': embedding_dim(self.num_classes['gender'])
                }
            },
            'text_features': {
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
    user_dataset = UserDataset()
    print(len(user_dataset))
    batch = [user_dataset[i] for i in user_dataset.dataframe.index[0:32]]
    print(batch)
    print(user_dataset.collate_fn(batch))