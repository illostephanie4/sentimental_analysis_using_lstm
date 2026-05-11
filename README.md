# Sentiment Analysis with LSTM

## Project Objective
This project aims to build a Sentiment Analysis model using a Long Short-Term Memory (LSTM) neural network to classify movie reviews from the IMDb dataset as Positive (1) or Negative (0).

## Methodology

### 1. Data Loading and Sampling
- The IMDb dataset is loaded using the `datasets` library.
- A sample of 10,000 training entries is extracted from the full training dataset using stratified sampling to ensure a balanced representation of positive and negative reviews.
- The sampled training data is then split into an 80% training set and a 20% validation set, also with stratification.

### 2. Text Processing
- **Tokenization:** Reviews are tokenized into individual words using NLTK's `word_tokenize`.
- **Vocabulary Creation:** A vocabulary of the 20,000 most common words is created, along with special tokens for padding (`<pad>`) and unknown words (`<unk>`).
- **Conversion to Integer Sequences:** Each tokenized review is converted into a sequence of integer IDs based on the created vocabulary.
- **Padding/Truncating:** Reviews are padded with `<pad>` tokens or truncated to a fixed length (`seq_len=300`) to ensure uniform input size for the LSTM model.

### 3. GloVe Embeddings
- Pre-trained GloVe embeddings (6B, 100d) are downloaded and loaded.
- An embedding matrix is created, mapping each word in the vocabulary to its corresponding GloVe vector. Words not found in GloVe are initialized randomly, and the padding token is set to all zeros.

### 4. LSTM Model Architecture
- A `SentimentLSTM` model is defined using PyTorch's `nn.Module`.
- The model consists of:
    - An `nn.Embedding` layer initialized with the GloVe embedding matrix (allowing fine-tuning).
    - An `nn.Dropout` layer for regularization.
    - An `nn.LSTM` layer with `n_layers=2`, `hidden_dim=512`, and `bidirectional=True` to capture contextual information from both directions.
    - A mean pooling operation applied to the LSTM output across the sequence length.
    - A final `nn.Linear` layer for binary classification.
    - A `nn.Sigmoid` activation function to output probabilities between 0 and 1.

### 5. Training and Evaluation
- The model is trained using `nn.BCELoss` as the criterion and `Adam` optimizer with a learning rate of `0.001` and `weight_decay`.
- The training process includes:
    - Iterating for a maximum of `200` epochs.
    - Early stopping with a patience of `20` epochs based on validation loss.
    - Gradient clipping to prevent exploding gradients.
    - The best model weights are saved based on the highest validation accuracy.
- Evaluation metrics include binary accuracy, precision, recall, and F1 score.

## Results

### Training Progress
The model was trained for several epochs, with metrics tracked for both training and validation sets. Early stopping was implemented to prevent overfitting.

- **Best Validation Accuracy Achieved:** 86.90%

### Performance on Test Set
After training, the best performing model (saved as `best_model.pt`) was evaluated on a held-out test set (10,000 samples from the IMDb test split).

- **Test Loss:** 0.379
- **Test Accuracy:** 84.49%
- **Test Precision:** 0.842
- **Test Recall:** 0.849
- **Test F1 Score:** 0.845

### Visualizations
- **Training vs. Validation Loss Plot:** Shows the convergence and potential overfitting during training.
- **Training vs. Validation Accuracy Plot:** Illustrates the performance trend on both sets.
- **Confusion Matrix:** Provides a detailed breakdown of correct and incorrect classifications on the test set.
