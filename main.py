from src.load_python import load_and_merge_data
from src.preprocess import clean_text

data = load_and_merge_data()

data['combined'] = data['title'] + " " + data['text']

data['clean_text'] = data['combined'] .apply(clean_text)
 
print(data[['label', 'clean_text']].head())
