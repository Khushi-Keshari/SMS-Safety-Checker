# train_model.py
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Preprocessing function (no NLTK tokenizer)
def transform_text(text):
    text = text.lower()
    # Replace URLs with token
    text = re.sub(r"http\S+|www\S+|https\S+", "url", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenize using split
    tokens = text.split()
    # Remove stopwords and stem
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Preprocess messages
df['transformed'] = df['message'].apply(transform_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['transformed'], df['label'], test_size=0.2, random_state=2)

# Pipeline with TF-IDF (n-grams) + Naive Bayes
sms_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,3))),
    ('model', MultinomialNB())
])

# Train
sms_pipeline.fit(X_train, y_train)

# Evaluate (optional)
print("✅ Model accuracy:", sms_pipeline.score(X_test, y_test))

# Save pipeline
with open('sms_pipeline.pkl', 'wb') as f:
    pickle.dump(sms_pipeline, f)

print("✅ Pipeline saved successfully as sms_pipeline.pkl!")
