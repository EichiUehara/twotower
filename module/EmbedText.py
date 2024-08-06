from torch import nn
from torch.nn import functional as F
import torch
from transformers import AutoModel

class EmbedText(nn.Module):
    def __init__(self, model_name):
        super(EmbedText, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
    
    def forward(self, tokenized_text):
        with torch.no_grad():
            outputs = self.model(
                input_ids=tokenized_text['input_ids'], 
                attention_mask=tokenized_text['attention_mask'])
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
        return cls_embeddings

if __name__ == '__main__':
    from module.Tokenizer import Tokenizer
    from module.process_text_feature import process_text_feature
    import time

    # Initialize the text embedding model and tokenizer
    embed_text = EmbedText('BAAI/bge-base-en-v1.5')
    tokenizer = Tokenizer('BAAI/bge-base-en-v1.5')

    # Batch of text features
    text_features = [
        '{"Package Dimensions": "7.1 x 5.5 x 3 inches; 2.38 Pounds", "UPC": "617390882781"}',
        '{"Item Form": "Powder", "Finish Type": "Shimmery", "Brand": "HUDABEAUTY", "Color": "Assorted", "Unit Count": "1 Count"}',
        '{"Manufacturer": "Apple", "ASIN": "B07XQXZXJC", "Model Number": "MWF82LL/A"}'
    ]
    
    # Tokenize the batch of text features
    start = time.time()
    tokenized_texts = tokenizer.tokenize(
        text_features, padding=True, truncation=True, max_length=512, return_tensors='pt'
    )
    print(tokenized_texts)
    print(tokenizer.decode(tokenized_texts['input_ids'][0]))
    print(tokenized_texts['input_ids'].shape) # (batch_size, max_length_of_token)
    print(tokenized_texts['attention_mask'].shape) # (batch_size, max_length_of_token)
    
    # Embed the batch of tokenized text features
    text_embeddings = embed_text(tokenized_texts)
    print(text_embeddings.shape) # (batch_size, num_hidden_states)
    print(f"taking: {time.time() - start} milliseconds")

    # Example other features
    other_features = torch.randn(3, 5)  # (batch_size, feature_dim)
    print(f"Other features size: {other_features.size()}")  # Should be (3, 5)

    # Concatenate features along the second dimension
    combined_features = torch.cat([other_features, text_embeddings], dim=1)
    print(f"Combined features size: {combined_features.size()}")  # Should be (3, 15)

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
    
    feedforward_network = FeedForwardNetwork(input_dim=768 + 5, hidden_dim=128, output_dim=64)

    # Pass the combined features through the feedforward network
    output = feedforward_network(combined_features)
    print(f"Output size: {output.size()}")  # Should be (3, 1)
