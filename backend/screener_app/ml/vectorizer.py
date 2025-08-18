# ml/vectorizer.py

from sklearn.feature_extraction.text import TfidfVectorizer
from  screener_app.ml.preprocessing import clean_text

class ResumeVectorizer:
    def __init__(self):
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000)  # top 5000 words

    def fit_transform(self, docs):
        """
        Fit TF-IDF on training resumes.
        """
        cleaned_docs = [clean_text(doc) for doc in docs]
        return self.vectorizer.fit_transform(cleaned_docs)

    def transform(self, docs):
        """
        Transform new resumes using trained TF-IDF.
        """
        cleaned_docs = [clean_text(doc) for doc in docs]
        return self.vectorizer.transform(cleaned_docs)
