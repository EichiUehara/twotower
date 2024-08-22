# Two Tower Recommendation Model

Reference implementation

    python -m train_amazon_review_ids_medium

    python -m train_amazon_review_ids_small

    python -m train_amazon_review_text_item_small

    python -m train_amazon_review_text_small

    python -m train_amazon_review_text_user_small

    python -m train_movie_lens

## Data Loader
Main Task Of the Data Loader it to define function applying to the batch for training.

## collate_fn

collate_fn is a function which apply some manupuration to batch data from data loader. Typically beneficial for applying tokenization since tokenization in advance take significant amount of time but applying tokenization one by one is inefficient.

## Dataset

Benefit of Dataset is isoration of the data source and data modeling from Machine Learning Model.

For two tower model, we define

    Interaction Dataset

    Relevance Dataset

    Observation Dataset

### Module

Module is unit of the logic of the solution.

Common logic for deep learning solution is below

1. Embedding for the features.

2. Normarization

3. Processing batch item

4. Text Tokenization.

5. Language Model Embedding

6. Apply Attenstion to Sequence

7. Embedding Retreval

8. Selecting Hyper parameters

### Layer

Layer is reusable components in the Deep Learning model.

I prepared two layers for isolation of the logic.

1. FeatureEmbeddingLayer

Serves for embedding of features for each Domain of Interaction and Obsearvation tower.

2. FeedForwardNetwork

Apply non-linear transformation function to the embedded feature matrix to solve the optimization problem e.g. minimizing classification loss.

### Model

Model is the entire architecture of the machine learning model.

It use DataLoader, Dataset, Layer, module and other custom logic to implement the solution.
