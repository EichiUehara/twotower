from sklearn.preprocessing import LabelEncoder
import torch

from model.TwoTowerModel import TwoTowerBinaryModel


if __name__ == '__main__':
    from module.Tokenizer import Tokenizer
    from dataset.amazon_review_base.UserDataset import UserDataset
    from dataset.amazon_review_base.ItemDataset import ItemDataset
    from dataset.amazon_review_base.ReviewDataset import ReviewDataset
    import time
    item_label_encoder = LabelEncoder()
    user_label_encoder = LabelEncoder()
    tokenizer = Tokenizer('BAAI/bge-base-en-v1.5')
    review_dataset = ReviewDataset('All_Beauty', item_label_encoder, user_label_encoder)
    item_label_encoder.fit(review_dataset.dataframe['parent_asin'].unique())
    user_label_encoder.fit(review_dataset.dataframe['user_id'].unique())
    user_dataset = UserDataset(
        'All_Beauty', tokenizer,
        item_label_encoder=item_label_encoder, user_label_encoder=user_label_encoder,
        max_history_length=10)
    item_dataset = ItemDataset('All_Beauty', tokenizer, item_label_encoder=item_label_encoder)
    model = TwoTowerBinaryModel(64, 10, 
            item_label_encoder, 
            user_label_encoder,
            review_dataset, user_dataset, item_dataset)
    train_dataloader, val_dataloader = model.get_data_loaders(review_dataset, 256, 0.8)
    # # load the model
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    