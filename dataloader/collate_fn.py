import torch

from module.process_history_feature import process_history_feature
from module.process_text_feature import process_text_feature
from module.process_text_history_feature import process_text_history_feature

def collate_fn(
    batch, 
    numerical_features, categorical_features, text_features, 
    history_features, text_history_features, 
    tokenizer, max_history_length):
    # Process numerical features
    numerical_features_tensor = torch.stack([torch.tensor([item[k] for item in batch], dtype=torch.float32) for k in numerical_features])
    
    # Process categorical features
    categorical_features = {k: torch.tensor([item[k] for item in batch], dtype=torch.long) for k in categorical_features}
    
    # Process text features
    text_features = {
        k: process_text_feature([item[k] for item in batch], tokenizer, padding=True, truncation=True, max_length=512, return_tensors='pt')
        for k in text_features
    }
    
    # Process history features
    history_features = {
        k: torch.stack([process_history_feature(item[k], max_history_length) for item in batch])
        for k in history_features
    }
    
    # Process text history features
    text_history_features = {
        k: process_text_history_feature(
            [item[k] for item in batch], 
            max_history_length, 
            tokenizer, padding='max_length', 
            truncation=True, max_length=512, 
            return_tensors='pt')
        for k in text_history_features
    }
    
    # Return the batch in a format compatible with the model
    return {
        'id': torch.tensor([item['id'] for item in batch], dtype=torch.long),
        'numerical_features': numerical_features_tensor,
        'categorical_features': categorical_features,
        'text_features': text_features,
        'history_features': history_features,
        'text_history_features': text_history_features
    }