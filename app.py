import streamlit as st
import joblib
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Download NLTK resources (only needed first time)
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer
ps = PorterStemmer()

# Function to preprocess text (same as in your Jupyter notebook)
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Streamlit UI
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #6C63FF;'>📧 Email / SMS Spam Classifier</h1>
    <p style='text-align: center; color: grey;'>Detect whether a message is Spam or Not Spam using Machine Learning</p>
    """,
    unsafe_allow_html=True,
)

# User input
input_sms = st.text_area("✉️ Enter the message you want to check:")

if st.button("🔍 Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message first!")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = vectorizer.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display result
        if result == 1:
            st.error("🚨 This message is **SPAM**!")
        else:
            st.success("✅ This message is **NOT SPAM**!")

        # Optional: Show cleaned text
        with st.expander("See processed message"):
            st.write(transformed_sms)


# Footer
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: #888;'>Made with ❤️ by Avinash</p>
    """,
    unsafe_allow_html=True,
)
