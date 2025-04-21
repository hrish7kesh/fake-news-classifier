from src.load_python import load_and_merge_data
from src.preprocess import clean_text
from src.train_model import train_model

data = load_and_merge_data()

data['combined'] = data['title'] + " " + data['text']

data['clean_text'] = data['combined'] .apply(clean_text)
 
model, X_test, y_test = train_model(data['clean_text'], data['label'])

print("Model training completed.")