# Twitter Sentiment Analysis Using Bidirectional LSTM

## Overview

This project implements a **Twitter Sentiment Analysis** model using a Bidirectional LSTM architecture built with TensorFlow and Keras. The goal is to classify tweets into multiple sentiment categories using deep learning techniques. The dataset consists of tweets labeled with various sentiments, which are used to train and evaluate the model.

## Features

- Text preprocessing (tokenization, padding)
- Embedding layer to convert tokens into dense vectors
- Bidirectional LSTM model for sequence processing
- Multi-class classification using softmax activation
- Early stopping to prevent overfitting
- Evaluation metrics: accuracy and confusion matrix

## Model Architecture

The model is designed using the following layers:
1. **Embedding Layer**: Converts tokens into 16-dimensional dense vectors.
2. **Bidirectional LSTM Layers**: Two stacked bidirectional LSTM layers with 20 units each.
3. **Dense Layer**: Output layer with 4 units, using softmax activation for multi-class classification.


## Dependencies

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- TensorFlow/Keras
- NLTK
- TextBlob
- Scikit-learn

## Dataset

The model is trained using a dataset of tweets labeled with sentiment classes. The dataset files (`twitter_training.csv` and `twitter_validation.csv`) should be placed in the `data/` directory.

### Example Dataset Format:
| Tweet_ID | Entity  | Sentiment | Tweet_Content      |
| -------- | ------- | --------- | ------------------ |
| 12345    | Product | Positive  | I love this phone! |
| 67890    | Brand   | Negative  | Worst service ever |

- `Tweet_ID`: Unique identifier for the tweet.
- `Entity`: The entity related to the tweet (e.g., product, service, brand).
- `Sentiment`: The sentiment associated with the tweet (e.g., Positive, Negative, Neutral).
- `Tweet_Content`: The content of the tweet.

## Usage

1. **Preprocessing Data**:
   The script will handle tokenization, padding, and text preprocessing using NLTK and TextBlob for stopword removal, tokenization, and lemmatization.

2. **Training the Model**:
   You can train the model by running the following command:
   ```bash
   python train_model.py
   ```

   The model is trained on the processed training data and evaluated on the test data.

3. **Evaluating the Model**:
   After training, the model will be evaluated using accuracy and a confusion matrix to determine performance. You can view these results in the output.

## Model Training

The model is trained using the following settings:
- **Loss function**: `sparse_categorical_crossentropy`
- **Optimizer**: `adam`
- **Metrics**: `accuracy`
- **Epochs**: 15
- **Callbacks**: Early stopping based on accuracy with a patience of 5 epochs.


