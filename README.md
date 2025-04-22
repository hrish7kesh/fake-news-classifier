# 📰 Fake News Detection Using NLP & Machine Learning

A machine learning pipeline to detect whether a news article is **real or fake**, built using NLP techniques and real-world news data.

## ✅ Features
- Real + Fake news datasets merged from Kaggle
- Text cleaning (lowercasing, stopwords, punctuation removal)
- TF-IDF vectorization for feature extraction
- Trained using PassiveAggressiveClassifier
- Model evaluation with accuracy, F1-score, and confusion matrix
- Clean, modular Python scripts

## 🧠 Tech Stack
- Python
- scikit-learn
- pandas, numpy
- nltk
- matplotlib, seaborn
- [Optional] Streamlit

## 🚀 How to Run

1. Clone the repo:
```bash
git clone https://github.com/hrish7kesh/fake-news-classifier.git
cd fake-news-classifier
```

2. Install dependencies:
pip install -r requirements.txt

3. Download the dataset from Kaggle:
https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets
Place Fake.csv and True.csv inside a folder called News_dataset/

4. Run the project:
python main.py

## 📈 Output Sample

- ✅ Accuracy: ~94%
- 📊 Classification report with precision, recall, and F1-score
- 🧩 Confusion matrix displayed using Seaborn heatmap
- 💾 Trained model saved as `models/fake_news_model.pkl`

---

## 🚧 Future Work

- 🔧 Add Streamlit interface for user input and live predictions
- 🤖 Experiment with transformer models (e.g., BERT via HuggingFace)
- 🌍 Extend to multilingual or multi-platform news datasets
