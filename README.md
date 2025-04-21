# 📰 Fake News Classifier

A machine learning project to detect fake vs real news using NLP and Python.  
Currently supports data loading, preprocessing, and training with TF-IDF and Passive Aggressive Classifier.

---

## ✅ Current Features
- ✅ Load and merge real/fake news datasets
- ✅ Clean and preprocess text (remove stopwords, punctuation, etc.)
- ✅ Train a basic ML classifier (TF-IDF + PassiveAggressiveClassifier)

---

## 🛠 Tech Stack
- Python
- Pandas
- NLTK
- Scikit-learn

---

## 🧠 Upcoming Features
- [ ] Evaluation metrics and confusion matrix
- [ ] Streamlit UI for live predictions
- [ ] Model improvement with BERT or LSTM
- [ ] Deployment to Streamlit Cloud or HuggingFace Spaces

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py

---


## 📥 Dataset

This project uses the [Fake and Real News Dataset from Kaggle](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets).  
Download the dataset and place 'fake.csv' and 'true.csv' inside the 'News_dataset/' folder.
