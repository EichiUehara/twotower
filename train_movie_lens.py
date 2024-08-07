from sklearn.preprocessing import LabelEncoder
import torch

from model.TwoTowerModel import TwoTowerBinaryModel


if __name__ == '__main__':
    from dataset.movie_lens.UserDataset import UserDataset
    from dataset.movie_lens.ItemDataset import ItemDataset
    from dataset.movie_lens.ReviewDataset import ReviewDataset
    review_dataset = ReviewDataset()
    user_dataset = UserDataset()
    item_dataset = ItemDataset()
    model = TwoTowerBinaryModel(64, 10, user_dataset, item_dataset)
    train_dataloader, val_dataloader = model.get_data_loaders(review_dataset, 512, 0.8, num_workers=8)
    
    # start_time = time.time()
    # for batch in train_dataloader:
    #     user_ids = batch['user_id']
    #     item_ids = batch['item_id']
    #     print(f"Review Data Time: {time.time() - start_time}")
    #     user_batch = [user_dataset[id] for id in user_ids]
    #     print(f"User Data Time: {time.time() - start_time}")
    #     user_batch = user_dataset.collate_fn(user_batch)
    #     print(f"User Data Batch Time: {time.time() - start_time}")
    #     item_batch = [item_dataset[id] for id in item_ids]
    #     print(f"Item Data Time: {time.time() - start_time}")
    #     item_batch = item_dataset.collate_fn(item_batch)
    #     print(f"Item Data Batch Time: {time.time() - start_time}")
    #     start_time = time.time()
    #     # Training code here
    #     pass

    model.fit(torch.optim.Adam(model.parameters(), lr=0.001), train_dataloader)
    # save the model
    torch.save(model.state_dict(), 'model.pth')
    # # load the model
    # model.load_state_dict(torch.load('model.pth'))
    # model.eval()
    