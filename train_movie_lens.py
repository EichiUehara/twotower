import torch
from model.TwoTowerModel import TwoTowerBinaryModel

import warnings
warnings.filterwarnings("ignore", message="User provided device_type of 'cuda', but CUDA is not available. Disabling")

if __name__ == '__main__':
    from dataset.movie_lens_ids.UserDataset import UserDataset
    from dataset.movie_lens_ids.ItemDataset import ItemDataset
    from dataset.movie_lens_ids.ReviewDataset import ReviewDataset
    review_dataset = ReviewDataset()
    user_dataset = UserDataset()
    item_dataset = ItemDataset()
    model = TwoTowerBinaryModel(256, 10, user_dataset, item_dataset)
    train_dataloader, val_dataloader = model.get_data_loaders(review_dataset, 2048, 0.8, num_workers=8)
    model.fit(torch.optim.Adam(model.parameters(), lr=0.001), train_dataloader, val_dataloader, epochs=5)
    # save the model
    torch.save(model.state_dict(), 'model.pth')
    # # load the model
    # model.load_state_dict(torch.load('model.pth'))
    # model.eval()
    