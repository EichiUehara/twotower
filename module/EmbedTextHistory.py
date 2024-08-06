from torch import nn
from torch.nn import functional as F
import torch
from transformers import AutoModel

class EmbedTextHistory(nn.Module):
    def __init__(self, model_name):
        super(EmbedTextHistory, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
    
    def forward(self, tokenized_text_histories):
        with torch.no_grad():
            outputs = self.model(
                input_ids=tokenized_text_histories['input_ids'], 
                attention_mask=tokenized_text_histories['attention_mask'])
            embeddings = outputs.last_hidden_state
        cls_embeddings_per_history = embeddings[:, 0, :]  # CLS token embeddings
        return cls_embeddings_per_history

if __name__ == '__main__':
    from module.Tokenizer import Tokenizer
    from module.process_text_history_feature import process_text_history_feature
    import time

    transformer_encoder = nn.TransformerEncoderLayer(d_model=768, nhead=16)
    embed_text_history = EmbedTextHistory('BAAI/bge-base-en-v1.5')
    tokenizer = Tokenizer('BAAI/bge-base-en-v1.5')

    # Sample batched text histories
    text_histories_batch = [["My dog is cute", "My cat is cute"], ["Bra Bra", "Foo Foo", "Bar Bar"]]
    max_history_length = 20
    
    # Process text history features
    start = time.time()
    tokenized_text_histories = process_text_history_feature(
        text_histories_batch, max_history_length, tokenizer, padding=True, truncation=True, max_length=512, return_tensors='pt')
    
    print(tokenized_text_histories['input_ids'].shape)  # (batch_size, num_histories, max_length_of_token)
    print(tokenized_text_histories['attention_mask'].shape)  # (batch_size, num_histories, max_length_of_token)
    
    # Forward pass through the embedding model
    embedded_histories = embed_text_history(tokenized_text_histories)
    print(embedded_histories.shape)  # (batch_size, embedding_dim)
    print(f"Taking: {time.time() - start} milliseconds")
    
    # other features
    other_features = torch.randn(2, 5)  # (batch_size, feature_dim)

    # Define a feedforward network
    class FeedForwardNetwork(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(FeedForwardNetwork, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    feedforward_network = FeedForwardNetwork(input_dim=(embedded_histories.size(1) + other_features.size(1)), hidden_dim=20, output_dim=1)

    # Pass the combined features through the feedforward network
    output = feedforward_network(torch.cat([embedded_histories, other_features], dim=1))
    print(f"Output size: {output.size()}")  # Should be (batch_size, 1)
