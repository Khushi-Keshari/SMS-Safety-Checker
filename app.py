import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# ✅ Load the trained pipeline (which already includes vectorizer + model)
with open('sms_pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Streamlit UI
st.title("📩 SMS Classifier (Spam Detector)")

# Use session_state to store the input value
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
