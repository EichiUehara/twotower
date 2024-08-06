def process_y_label(user_features, item_features):
    purchase_items = user_features['verified_purchase']
    item_id = item_features['parent_asin']
    if item_id in purchase_items:
        return 1
    return 0

if __name__ == '__main__':
    user_features = {
        'verified_purchase': ['B00FALQ1ZC', 'B00FALQ1ZD']
    }
    item_features = {
        'parent_asin': 'B00FALQ1ZC'
    }
    print(process_y_label(user_features, item_features))
    item_features = {
        'parent_asin': 'B00FALQ1ZD'
    }
    print(process_y_label(user_features, item_features))