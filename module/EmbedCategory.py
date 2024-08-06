from torch import nn
import torch

class EmbedCategory(nn.Module):
    def __init__(self, num_class, embedding_dim):
        super(EmbedCategory, self).__init__()
        self.embedding_layer = nn.Embedding(num_class, embedding_dim)

    def forward(self, feature):
        # Assumes feature is a tensor of shape (batch_size,)
        return self.embedding_layer(feature)

if __name__ == '__main__':
    from torch import nn
    from torch.nn import functional as F
    batch_size = 4

    class FeedForwardNetwork(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(FeedForwardNetwork, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    # Initialize the embedding layer and feed-forward network
    embed_category = EmbedCategory(num_class=10, embedding_dim=5) # 10 classes, output feature dimension of 5
    category_features = torch.tensor([1, 2, 3, 4])  # (batch_size,)
    other_features = torch.randn(batch_size, 10)  # (batch_size, feature_dim)
    feedforward_network = FeedForwardNetwork(input_dim=(embed_category.embedding_layer.embedding_dim + other_features.size(1)), hidden_dim=128, output_dim=64)
    # Get embeddings for the categorical features
    category_embeddings = embed_category(category_features)
    print(f"Category embeddings shape: {category_embeddings.shape}")

    # Pass embeddings and other features through the feed-forward network
    output = feedforward_network(torch.cat([category_embeddings, other_features], dim=1))
    print(f"Output shape: {output.shape}")
