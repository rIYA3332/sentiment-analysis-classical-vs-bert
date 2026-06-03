# Sentiment Analysis: Logistic Regression vs DistilBERT

A binary sentiment classification project that compares a classical 
machine learning approach with a transformer-based model on movie reviews.

## Dataset
- Source: Stanford Sentiment Treebank (SST) via HuggingFace
- Task: Predict whether a review is Positive (1) or Negative (0)
- Neutral labels were removed and remaining labels converted to binary

## Models Used
- Logistic Regression with TF-IDF features
- DistilBERT fine-tuned for sequence classification

## Results

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | ~75%     |
| DistilBERT          | ~88%     |

DistilBERT outperformed Logistic Regression across all metrics, 
showing the benefit of contextual embeddings over traditional 
bag-of-words methods.

## What the notebook covers
- Text cleaning and lemmatization
- TF-IDF vectorization with bigrams
- DistilBERT fine-tuning with early stopping
- Hyperparameter tuning using GridSearchCV
- Evaluation using Accuracy, Precision, Recall, F1, ROC Curve, 
  and Confusion Matrix

## Libraries
Python, PyTorch, HuggingFace Transformers, Scikit-learn, 
NLTK, Pandas, Matplotlib, Seaborn
