from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)
    def tokenize(self, text, max_length=512, 
               padding='max_length', truncation=True, 
               return_tensors='pt'):
        tokenized = self.tokenizer(text, padding=padding,
                                        truncation=truncation, 
                                        max_length=max_length, 
                                        return_tensors=return_tensors)
        return tokenized
    def encode(self, text, max_length=512, 
               padding='max_length', truncation=True, 
               return_tensors='pt'):
        return self.tokenizer.encode(text, padding=padding, 
                                     truncation=truncation, 
                                     max_length=max_length, 
                                     return_tensors=return_tensors)
    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)

if __name__ == '__main__':
    tokenizer = Tokenizer("BAAI/bge-base-en-v1.5")
    print(tokenizer.tokenize("Hello, my dog is cute")["input_ids"])
    encoded = tokenizer.encode("Hello, my dog is cute")
    print(encoded)
    print(tokenizer.decode(encoded.squeeze(0)))