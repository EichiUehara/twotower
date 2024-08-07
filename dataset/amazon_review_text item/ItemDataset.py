from dataset.amazon_review.ItemDataset import ItemDataset
class ItemDataset(ItemDataset):
    def __init__(self, category):
        super().__init__(category)
        self.numerical_features = ['average_rating']
        self.categorical_features = []
        self.text_features = []
        self.history_features = ['purchased_item_ids']
        self.text_history_features = ['review_text_history']

if __name__ == '__main__':
    from sklearn.preprocessing import LabelEncoder
    item_label_encoder = LabelEncoder()
    item_dataset = ItemDataset('All_Beauty')
    item_dataset.item_label_encoder.fit(item_dataset.dataframe.index)
    print(len(item_dataset))