import streamlit as st
import joblib
import nltk
import string
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', download_dir=nltk_data_dir)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)



ps = PorterStemmer()

model = joblib.load("spam_classifier_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


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


st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #6C63FF;'>📧 Email / SMS Spam Classifier</h1>
    <p style='text-align: center; color: grey;'>Detect whether a message is Spam or Not Spam using Machine Learning</p>
    """,
    unsafe_allow_html=True,
)

input_sms = st.text_area("✉️ Enter the message you want to check:")

if st.button("🔍 Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message first!")
    else:
        transformed_sms = transform_text(input_sms)

        vector_input = vectorizer.transform([transformed_sms])

        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚨 This message is **SPAM**!")
        else:
            st.success("✅ This message is **NOT SPAM**!")

        with st.expander("See processed message"):
            st.write(transformed_sms)


st.markdown(
    """
    <hr>
    <p style='text-align: center; color: #888;'>Made with ❤️</p>
    """,
    unsafe_allow_html=True,
)




