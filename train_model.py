import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# ✅ Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# ✅ Load and clean dataset
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Apply preprocessing to training data
df['transformed'] = df['message'].apply(transform_text)

# Split
X_train, X_test, y_train, y_test = train_test_split(df['transformed'], df['label'], test_size=0.2, random_state=2)

# ✅ Create pipeline for TF-IDF + model
sms_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=3000)),
    ('model', MultinomialNB())
])

# Train
sms_pipeline.fit(X_train, y_train)

# ✅ Save entire pipeline
with open('sms_pipeline.pkl', 'wb') as f:
    pickle.dump(sms_pipeline, f)

print("✅ Pipeline (vectorizer + model) saved successfully as sms_pipeline.pkl!")
