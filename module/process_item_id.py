def process_item_id(item_ids, item_id_to_index):
    item_indexes = []
    for item_id in item_ids:
        if item_id_to_index.get_item_index(item_id) is None:
            item_id_to_index.add(item_id)
        item_indexes.append(item_id_to_index.get_item_index(item_id))
    
    return item_indexes


if __name__ == '__main__':
    import pandas as pd
    import uuid
    from ItemIdToIndex import ItemIdToIndex
    random_uuids = [str(uuid.uuid4()) for _ in range(100)]
    pd_series = pd.Series(random_uuids)
    item_id_to_index = ItemIdToIndex(pd_series)
    item_ids = random_uuids[:10]
    item_indexes = process_item_id(item_ids, item_id_to_index)
    print(item_indexes)
    item_ids = random_uuids[:10] + ['test']
    item_indexes = process_item_id(item_ids, item_id_to_index)
    print(item_indexes)
    print(item_id_to_index.get_item_index('test'))
    item_ids = random_uuids[:10] + ['test1']
    item_indexes = process_item_id(item_ids, item_id_to_index)
    print(item_indexes)
    print(item_id_to_index.get_item_index('test1'))
    item_ids = random_uuids[:10] + ['test2']
    item_indexes = process_item_id(item_ids, item_id_to_index)
    print(item_indexes)
    print(item_id_to_index.get_item_index('test3'))