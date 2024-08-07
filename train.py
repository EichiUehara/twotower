from sklearn.preprocessing import LabelEncoder
import torch

from model.TwoTowerModel import TwoTowerBinaryModel


if __name__ == '__main__':
    from module.Tokenizer import Tokenizer
    from dataset.amazon_review.UserDataset import UserDataset
    from dataset.amazon_review.ItemDataset import ItemDataset
    from dataset.amazon_review.ReviewDataset import ReviewDataset
    import time
    item_label_encoder = LabelEncoder()
    user_label_encoder = LabelEncoder()
    tokenizer = Tokenizer()
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
    train_dataloader, val_dataloader = model.get_data_loaders(review_dataset, 128, 0.8, num_workers=8)
    
    start_time = time.time()
    for batch in train_dataloader:
        user_ids = batch['user_id']
        item_ids = batch['item_id']
        print(f"Review Data Time: {time.time() - start_time}")
        user_batch = [user_dataset[id] for id in user_ids]
        print(f"User Data Time: {time.time() - start_time}")
        user_batch = user_dataset.collate_fn(user_batch)
        print(f"User Data Batch Time: {time.time() - start_time}")
        item_batch = [item_dataset[id] for id in item_ids]
        print(f"Item Data Time: {time.time() - start_time}")
        item_batch = item_dataset.collate_fn(item_batch)
        print(f"Item Data Batch Time: {time.time() - start_time}")
        start_time = time.time()
        # Training code here
        pass

    model.fit(torch.optim.Adam(model.parameters(), lr=0.001), train_dataloader)
    # save the model
    torch.save(model.state_dict(), 'model.pth')
    # # load the model
    # model.load_state_dict(torch.load('model.pth'))
    # model.eval()
    