import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torch import nn
import torch

from layer.FeedForwardNetwork import FeedForwardNetwork
from module.EmbedCategory import EmbedCategory
from module.EmbedHistory import EmbedHistory
from module.NormNumeric import NormNumeric
from module.EmbedText import EmbedText
from module.EmbedTextHistory import EmbedTextHistory

class FeatureEmbeddingLayer(nn.Module):
    def __init__(self, embedding_dim, dataset):
        super(FeatureEmbeddingLayer, self).__init__()
        self.id_embedding = nn.Embedding(len(dataset), 200)
        self.embed_categorical = EmbedCategory(200000, 50)
        self.embed_history = EmbedHistory(
            nn.Embedding(200000, 50),
            nn.TransformerEncoderLayer(d_model=50, nhead=2))
        self.embed_numerical = NormNumeric(len(dataset.numerical_features))
        self.embed_text = EmbedText("BAAI/bge-base-en-v1.5")
        self.embed_text_history = EmbedTextHistory(
            "BAAI/bge-base-en-v1.5"
        )
        self.dataset = dataset
        self.output = FeedForwardNetwork(dataset.input_dim, 128, embedding_dim)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.output = nn.Linear(dataset.input_dim, embedding_dim)

    def forward(self, ids)->torch.Tensor:
        batch = [self.dataset[id] for id in ids]
        batch = self.dataset.collate_fn(batch)
        embedded_features = []
        embedded_features.append(self.id_embedding(batch['id'].to(self.device)))
        if len(batch['numerical_features']) > 0:
            embedded_features.append(
                self.embed_numerical(batch['numerical_features'].to(self.device)))
        if len(batch['categorical_features']) > 0:
            for feature in batch['categorical_features']:
                embedded_features.append(
                    self.embed_categorical(batch['categorical_features'][feature].to(self.device)))
        if len(batch['history_features']) > 0:
            for feature in batch['history_features']:
                embedded_features.append(
                    self.embed_history(batch['history_features'][feature].to(self.device)))
        if len(batch['text_features']) > 0:
            for feature in batch['text_features']:
                embedded_features.append(
                    self.embed_text(batch['text_features'][feature].to(self.device)))
        if len(batch['text_history_features']) > 0:
            for feature in batch['text_history_features']:
                embedded_features.append(
                    self.embed_text_history(batch['text_history_features'][feature].to(self.device)))
        concatenated = torch.cat(embedded_features, dim=1)
        return self.output(concatenated)
    
if __name__ == '__main__':
    from module.Tokenizer import Tokenizer
    from dataset.amazon_review_base.UserDataset import UserDataset
    from sklearn.preprocessing import LabelEncoder
    from dataset.amazon_review_base.ItemDataset import ItemDataset
    import time
    item_label_encoder = LabelEncoder()
    user_label_encoder = LabelEncoder()
    tokenizer = Tokenizer()
    user_dataset = UserDataset(
        'All_Beauty', tokenizer,
        item_label_encoder=item_label_encoder, user_label_encoder=user_label_encoder,
        max_history_length=10)
    user_dataset.user_label_encoder.fit(user_dataset.dataframe.index)
    item_dataset = ItemDataset('All_Beauty', tokenizer, item_label_encoder=item_label_encoder)
    item_dataset.item_label_encoder.fit(item_dataset.dataframe.index)
    # User_embedding = FeatureEmbeddingLayer(user_dataset.input_dim, 64, user_dataset)
    # ids = [user_dataset[i]['id'] for i in range(32)]
    # start = time.time()
    # print(User_embedding(ids).shape)
    # print(time.time() - start)
    item_embedding = FeatureEmbeddingLayer(item_dataset.input_dim, 64, item_dataset)
    ids = [item_dataset[i]['id'] for i in range(32)]
    start = time.time()
    print(item_embedding(ids).shape)
    print(time.time() - start)