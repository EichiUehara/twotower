import time
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
# from torch.utils.tensorboard import SummaryWriter
scaler = GradScaler()

from dataloader.get_data_loaders import get_data_loaders
from layer.FeatureEmbeddingLayer import FeatureEmbeddingLayer
from module.FaissIndex import FaissIndex


class TwoTowerBinaryModel(nn.Module):
    def __init__(self, embedding_dim, num_faiss_clusters, UserDataset, ItemDataset, model_name="BAAI/bge-base-en-v1.5"):
        super(TwoTowerBinaryModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.user_features_embedding = FeatureEmbeddingLayer(embedding_dim, UserDataset, model_name=model_name).to(self.device)
        self.item_features_embedding = FeatureEmbeddingLayer(embedding_dim, ItemDataset, model_name=model_name).to(self.device)
        self.FaissIndex = FaissIndex(embedding_dim, num_faiss_clusters)
        self.get_data_loaders = get_data_loaders
        # self.writer = SummaryWriter(log_dir=f"./runs/{UserDataset.__class__.__name__}_vs_{ItemDataset.__class__.__name__}_embedding_dim_{embedding_dim}".replace(' ', ''))

    def forward(self, user_ids, item_ids):
        user_emb = self.user_features_embedding(user_ids)
        item_emb = self.item_features_embedding(item_ids)
        interaction_score = torch.sum(user_emb * item_emb, dim=1)
        return interaction_score

    def index_train(self, data_loader):
        item_ids = []
        item_embs = []
        with torch.no_grad():
            for batch in data_loader:
                item_features = batch['item_id']
                item_emb = self.item_features_embedding(item_features)
                item_ids.append(item_features)
                item_embs.append(item_emb)
        item_ids = torch.cat(item_ids)
        item_embs = torch.cat(item_embs)
        self.FaissIndex.train(item_embs.cpu().numpy(), item_ids.cpu().numpy())
    
    def index_add(self, embedding, item_ids):
        self.FaissIndex.add(embedding, item_ids)
    
    def index_search(self, user_ids, topk):
        with torch.no_grad():
            user_emb = self.user_features_embedding(user_ids)
            _ , indices = self.FaissIndex.search(user_emb, topk)
            return [list(set(indices[i])) for i in range(len(indices))]
    
    def fit(self, optimizer, data_loader, val_data_loader=None, epochs=5):
        self.train()
        self.to(self.device)
        print("Hello World")
        print(f"Training on {self.device}")
        for epoch in range(epochs):
            start = time.time()
            i = 0
            running_loss = 0.0
            running_accuracy = 0.0
            for batch in data_loader:
                user_features = batch['user_id']
                item_features = batch['item_id']
                labels = batch['rating']
                loss, accuracy = self.train_step(optimizer, user_features, item_features, labels)
                i += 1
                running_loss += loss
                running_accuracy += accuracy
                if i % 100 == 0:
                    print(f"Loss: {running_loss / i:.4f}, Accuracy: {running_accuracy / i * 100:.2f}%")
                    print(f"Time: {time.time() - start}")
                    start = time.time()
                    print(f"Batches: {i}")
                    # self.writer.add_scalar('Loss', running_loss / i, epoch)
            if val_data_loader:
                self.evaluate(val_data_loader)

    def train_step(self, optimizer, user_features, item_features, labels):
        labels = labels.float().to(self.device)
        optimizer.zero_grad()
        with autocast():
            logits = self.forward(user_features, item_features)
            loss = F.binary_cross_entropy_with_logits(logits, labels)        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        probabilities = torch.sigmoid(logits)
        predictions = probabilities > 0.5
        accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        # print(f"Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%")
        return loss.item(), accuracy
    
    def evaluate(self, val_dataloader):
        self.eval()
        val_running_loss = 0.0
        val_running_accuracy = 0.0
        i = 0
        with torch.no_grad():
            for batch in val_dataloader:
                user_features = batch['user_id']
                item_features = batch['item_id']
                labels = batch['rating']
                logits = self.forward(user_features, item_features)
                probabilities = torch.sigmoid(logits)
                predictions = probabilities > 0.5
                accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
                val_running_loss += F.binary_cross_entropy_with_logits(logits, labels.float().to(self.device)).item()
                val_running_accuracy += accuracy
                i += 1
        print(f"Validation Loss: {val_running_loss / i:.4f}, Validation Accuracy: {val_running_accuracy / i * 100:.2f}%")
        # self.writer.add_scalar('Validation Loss', val_running_loss / i, i)
        # self.writer.add_scalar('Validation Accuracy', val_running_accuracy / i * 100, i)
                

    def inference(self, user_features, topk):
        return self.index_search(user_features, topk)