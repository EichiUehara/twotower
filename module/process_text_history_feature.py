import torch

from module.Tokenizer import Tokenizer

def process_text_history_feature(
    text_history_feature, max_history_length, tokenizer: Tokenizer, 
    padding='max_length', truncation=True, max_length=512, return_tensors='pt'
    ):
    new_text_history_feature = []
    for texts in text_history_feature:
        texts = texts[-max_history_length:]
        texts = [text for text in texts if text is not None and len(text) > 0]
        texts = "|".join(texts)
        new_text_history_feature.append(texts)
    print(new_text_history_feature)
    # new_text_history_feature = "|".join(["|".join(texts[-max_history_length:]) for texts in text_history_feature])
    return tokenizer.tokenize(
        new_text_history_feature,
        padding=padding, 
        truncation=truncation, 
        max_length=max_length, 
        return_tensors=return_tensors
    )


if __name__ == '__main__':
    tokenizer = Tokenizer()
    text_history_feature = [
        ['hello' * 10, 'world' * 10], 
        ['foo' * 10], 
        ['baz' * 10]
    ]
    max_history_length = 2
    tokenized = process_text_history_feature(text_history_feature, max_history_length, tokenizer)
    print(tokenized)
    print(tokenized["input_ids"])
    print(tokenized["attention_mask"])
    