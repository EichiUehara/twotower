from dataset.amazon_review.ItemDataset import ItemDataset
from dataset.amazon_review.UserDataset import UserDataset
class UserDataset(UserDataset):
    def __init__(self, category):
        super().__init__(category)
        self.numerical_features = ['average_rating', 'rating_number']
        self.categorical_features = ['main_category', 'store']
        self.text_features = ['details']
        self.history_features = []
        self.text_history_features = []

if __name__ == '__main__':
    item_dataset = ItemDataset('All_Beauty')
    user_dataset = UserDataset('All_Beauty', item_dataset.item_label_encoder)
    print(len(user_dataset))