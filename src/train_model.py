from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_df=0.7)),
        ('clf', PassiveAggressiveClassifier(max_iter=1000))
    ])

    # Train
    model.fit(X_train, y_train)

    return model, X_test, y_test
