import streamlit as st
import nltk
import re
import os
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -----------------------------
# ✅ Setup NLTK for Streamlit
# -----------------------------
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.mkdir(nltk_data_path)

nltk.data.path.append(nltk_data_path)

# Download required NLTK data quietly
nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)

# -----------------------------
# ✅ Text preprocessing
# -----------------------------
ps = PorterStemmer()

def transform_text(text):
    """
    Preprocess input text:
    - Lowercase
    - Tokenize
    - Remove punctuation & stopwords
    - Apply stemming
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "url", text)
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum() or i in "%$!"]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# -----------------------------
# ✅ Load trained pipeline
# -----------------------------
with open('sms_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# -----------------------------
# ✅ Streamlit UI
# -----------------------------
st.title("📩 SMS Classifier (Spam Detector)")

# Use session_state to store input
if "input_sms" not in st.session_state:
    st.session_state.input_sms = ""

input_sms = st.text_area("Enter the message", value=st.session_state.input_sms, height=100)

if st.button('Check'):
    if input_sms:
        try:
            # 1. Preprocess
            transformed_sms = transform_text(input_sms)
            # 2. Predict using pipeline directly
            result = pipeline.predict([transformed_sms])[0]
            # 3. Display result
            if result == 1:
                st.error("🚨 Spam Message Detected!")
            else:
                st.success("✅ This looks like a normal message.")

            # ✅ Reset the text area after checking
            st.session_state.input_sms = ""
           
        except Exception as e:
            st.error(f"Error processing input: {e}")
    else:
        st.warning("Please enter a message for prediction.")
