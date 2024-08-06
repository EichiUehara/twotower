import time
from sklearn.preprocessing import LabelEncoder
from torch.nn import functional as F
import torch
from torch import nn

from dataloader.get_data_loaders import get_data_loaders
from layer.FeatureEmbeddingLayer import FeatureEmbeddingLayer
from module.FaissIndex import FaissIndex

class TwoTowerBinaryModel(nn.Module):
    def __init__(self, 
                 embedding_dim, num_faiss_clusters, 
                 item_label_encoder: LabelEncoder, user_label_encoder: LabelEncoder,
                 ReviewDataset, UserDataset, ItemDataset):
        super(TwoTowerBinaryModel, self).__init__()
        self.user_features_embedding = FeatureEmbeddingLayer(embedding_dim, UserDataset)
        self.item_features_embedding = FeatureEmbeddingLayer(embedding_dim, ItemDataset)
        self.item_label_encoder = item_label_encoder
        self.item_label_encoder.fit(ItemDataset.dataframe.index)
        self.user_label_encoder = user_label_encoder
        self.user_label_encoder.fit(UserDataset.dataframe.index)
        self.review_dataset = ReviewDataset
        self.user_dataset = UserDataset
        self.item_dataset = ItemDataset
        self.FaissIndex = FaissIndex(embedding_dim, num_faiss_clusters)
        self.get_data_loaders = get_data_loaders
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, user_ids, item_ids):
        user_emb = self.user_features_embedding(user_ids)
        item_emb = self.item_features_embedding(item_ids)
        interaction_score = torch.sum(user_emb * item_emb, dim=1)
        interaction_prob = torch.sigmoid(interaction_score)
        return interaction_prob

    def index_train(self, item_ids):
        with torch.no_grad():
            item_emb = self.item_features_embedding(item_ids)
            self.FaissIndex.train(item_emb)
    
    def index_add(self, embedding, item_ids):
        self.FaissIndex.add(embedding, item_ids)
    
    def index_search(self, user_ids, topk):
        with torch.no_grad():
            user_emb = self.user_features_embedding(user_ids)
            _ , indices = self.FaissIndex.search(user_emb, topk)
            return indices
    
    def fit(self, optimizer, data_loader):
        print(f"Training on {self.device}")
        self.to(self.device)
        self.train()
        start = time.time()
        i = 0
        for batch in data_loader:
            user_features = batch['user_id']
            item_features = batch['item_id']
            labels = batch['rating'].to(self.device)
            loss = self.train_step(optimizer, user_features, item_features, labels)
            i += 1
            if i % 10 == 0:
                print(f"Loss: {loss}")
                print(f"Time: {time.time() - start}")
                start = time.time()
                print(f"Batches: {i}")

    def train_step(self, optimizer, user_features, item_features, labels):
        interaction_prob = self.forward(user_features, item_features)
        labels = labels.float()
        loss = F.binary_cross_entropy(interaction_prob, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def inference(self, user_features, topk):
        return self.index_search(user_features, topk)


if __name__ == '__main__':
    from module.Tokenizer import Tokenizer
    from dataset.amazon_review.UserDataset import UserDataset
    from dataset.amazon_review.ItemDataset import ItemDataset
    from dataset.amazon_review.ReviewDataset import ReviewDataset
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
    user_ids = [review_dataset[i]['user_id'] for i in range(32)]
    item_ids = [review_dataset[i]['item_id'] for i in range(32)]
    start = time.time()
    print(model(user_ids, item_ids))
    print(time.time() - start)
    # model.index_add(item_ids, item
    # model.index_train()
    # print(model.index_search(ids, 10))
    # model.fit(torch.optim.Adam(model.parameters()), get_data_loaders(user_dataset, item_dataset, review_dataset, 32))
    # print(model.inference(ids, 10))