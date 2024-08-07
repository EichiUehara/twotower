from module.Tokenizer import Tokenizer

def process_text_feature(
    text_feature, tokenizer: Tokenizer, 
    padding=True, truncation=True, max_length=512, return_tensors='pt'):
    return tokenizer.tokenize(
        text_feature, padding=padding, truncation=truncation, 
        max_length=max_length, return_tensors=return_tensors)

if __name__ == '__main__':
    tokenizer = Tokenizer("BAAI/bge-base-en-v1.5")
    text_feature = "My dog is cute"
    tokenized = process_text_feature(text_feature, tokenizer)
    print(tokenized["input_ids"])
    print(tokenized["attention_mask"])
    