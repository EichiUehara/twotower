import pandas as pd
import uuid


class ItemIdToIndex:
    def __init__(self):
        self.item_id_to_index = {}
        self.index_to_item = {}
        self.embedding_to_item = {}
    
    def __getitem__(self, idx):
        return self.index_to_item[idx]
    
    def get_item_index(self, item_id):
        if item_id in self.item_id_to_index:
            return self.item_id_to_index[item_id]
        return None

    def get_id_by_embedding(self, embedding):
        return self.index_to_item[embedding]
    
    def __len__(self):
        return len(self.item_id_to_index)

    def add(self, item_id):
        if item_id not in self.item_id_to_index:
            self.item_id_to_index[item_id] = len(self.item_id_to_index)
            self.index_to_item[len(self.item_id_to_index)] = item_id
            return True
        return False

    def add_embedding(self, item_id, embedding):
        self.embedding_to_item[embedding] = item_id
        return True
    

if __name__ == '__main__':
    item_id_to_index = ItemIdToIndex()
    print(item_id_to_index.add('test'))
    print(item_id_to_index.add('test'))
    print(item_id_to_index.get_item_index('test'))
    print(item_id_to_index.add('test1'))
    print(item_id_to_index.get_item_index('test1'))
    print(item_id_to_index.add('test2'))
    print(item_id_to_index.get_item_index('test3'))
    print(len(item_id_to_index))
    for i in range(100000):
        item_id_to_index.add(str(uuid.uuid4()))
    for i in range(100000):
        item_id_to_index.add_embedding(str(uuid.uuid4()), i)
        
    print(len(item_id_to_index))
    print(item_id_to_index.get_id_by_embedding(1))