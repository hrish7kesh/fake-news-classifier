import pandas as pd
from pathlib import Path

def load_and_merge_data():
    fake_path = Path("News_dataset/Fake.csv")
    true_path = Path("News_dataset/True.csv")

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df['label'] = 'FAKE'
    true_df['label'] = 'REAL'

    merged_df = pd.concat([fake_df, true_df], ignore_index=True)

    return merged_df