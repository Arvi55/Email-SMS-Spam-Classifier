import streamlit as st
import joblib
import numpy as np

# Load saved model and vectorizer
model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Streamlit UI setup
st.set_page_config(page_title="Email/SMS Spam Classifier", page_icon="📩", layout="centered")

st.title("📩 Email/SMS Spam Classifier")
st.markdown("### Classify your messages instantly using Machine Learning!")

st.write("Type or paste a message below to check if it's spam or not:")

# Text input area
user_input = st.text_area("Enter your message here:")

# Predict button
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        # Transform and predict
        input_features = vectorizer.transform([user_input])
        prediction = model.predict(input_features)[0]
        probability = model.predict_proba(input_features)[0]

        # Display result
        if prediction == 1:
            st.error(f"🚨 This message is classified as **SPAM**! (Confidence: {np.max(probability)*100:.2f}%)")
        else:
            st.success(f"✅ This message is **NOT SPAM**. (Confidence: {np.max(probability)*100:.2f}%)")

st.markdown("---")
st.caption("Built with ❤️ by Avinash using Streamlit & Machine Learning")
