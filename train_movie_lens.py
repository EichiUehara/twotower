import math
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
    model = TwoTowerBinaryModel(64, int(math.sqrt(len(item_dataset))), user_dataset, item_dataset)
    train_dataloader, val_dataloader = model.get_data_loaders(review_dataset, 2048, 0.8, num_workers=12)
    model.fit(torch.optim.Adam(model.parameters(), lr=0.001), train_dataloader, val_dataloader, epochs=32)
    # save the model
    torch.save(model.state_dict(), 'movie_lens.pth')
    # load the model
    model.load_state_dict(torch.load('movie_lens.pth'))
    # recommend for a user
    model.index_train(data_loader=train_dataloader)
    user_ids = next(iter(train_dataloader))['user_id'][:10]
    recommended_item_ids = model.index_search(user_ids, 30)
    for user_id, item_ids in zip(user_ids, recommended_item_ids):
        print(f"Recommended items: {item_ids} for user: {user_id.item()}")
        for item_id in item_ids:
            print(item_dataset.dataframe.loc[item_id])
            # Ensure embeddings are normalized before calculating similarity
            user_emb = torch.nn.functional.normalize(model.user_features_embedding(torch.tensor([user_id.item()])), dim=1)
            item_emb = torch.nn.functional.normalize(model.item_features_embedding(torch.tensor([item_id])), dim=1)

            similarity = torch.nn.functional.cosine_similarity(user_emb, item_emb, dim=1)
            print(f"Cosine Similarity: {similarity.item()}")
            print("dot product:", torch.mm(user_emb, item_emb.T).item())
            # print("dot product:", torch.mm(user_emb, item_emb.T).item() / (torch.norm(user_emb) * torch.norm(item_emb)))