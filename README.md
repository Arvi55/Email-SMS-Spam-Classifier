# 📧 Spam Email Classifier using Machine Learning  

🚀 **Live Project:** [Click Here to Try the Spam Classifier](https://email-sms-spam-classifier-avinash.streamlit.app/) 

![Web Face](https://github.com/Arvi55/Email-SMS-Spam-Classifier/blob/main/images/web%20face.png?raw=true)

---

##  Overview
This project classifies emails into **Spam** or **Ham (Not Spam)** using **Machine Learning (ML)** and **Natural Language Processing (NLP)** techniques.  
Spam messages waste storage and time and can expose users to phishing and scams.  
Our goal was to create a **smart email classifier** that learns patterns in spam messages and predicts them accurately.

---

## 🧩 Problem Statement
Traditional spam filters rely on fixed rules or keyword matching — which fail as spammers adapt their language.  
We aimed to:
- Build a **learning-based model** to identify spam automatically.  
- Compare the performance of **Naive Bayes classifiers**.  
- Create a model that can be deployed for real-time spam detection.

---

## 📂 Dataset Information

- **Dataset Used:** [Spam.csv Dataset](https://github.com/Arvi55/Email-SMS-Spam-Classifier/blob/main/spam.csv)
- **Shape:** (5572, 2)
- **Columns:**
  - `label`: Spam or Ham  
  - `text`: Message content  

---

## 🔍 Initial Data Structure & Checks

Before model training, several data checks were performed:
- Missing values check ✅  
- Duplicate detection ✅  
- Label balance analysis ✅  
- Basic word and sentence length analysis ✅  

### 🧾 Sample Dataset
![Data Head](https://github.com/Arvi55/Email-SMS-Spam-Classifier/blob/main/images/Data%20preview.png?raw=true)

### 📊 Label Distribution
- Ham: 4825  
- Spam: 747  

*(Dataset imbalance handled via evaluation metrics like precision and recall)*

---

## 🧮 Data Preprocessing

### Steps:
1. Convert text to lowercase  
2. Remove punctuation, symbols, and numbers  
3. Tokenize text  
4. Remove stopwords  
5. Lemmatize tokens  

---

## 🧩 Model Building and Training

After preprocessing the data and transforming text into numerical features using TF-IDF, multiple Machine Learning models were trained and evaluated to determine the most effective spam detection algorithm.

### 🧠 Models Used
- **Gaussian Naive Bayes**
- **Multinomial Naive Bayes**
- **Bernoulli Naive Bayes**

Each model was tested using an 80-20 train-test split.  
Evaluation metrics such as Accuracy, Precision and Confusion Matrix were used to assess performance.

---

## 📈 Model Evaluation and Results

| Model | Accuracy | Precision |
|--------|-----------|------------|
| GaussianNB | 90.03% | 0.60 | 
| MultinomialNB | **95.55%** | **1.00** | 
| BernoulliNB | 93.12% | 0.91 | 

**Multinomial Naive Bayes** performed best, achieving the highest accuracy and perfect precision, making it ideal for text-based spam detection.

---

## 🧮 Confusion Matrix

The confusion matrix highlights the number of correctly and incorrectly classified samples for each class.

![Confusion Matrix]()

**Interpretation:**
- **True Positives (TP):** Spam messages correctly identified as spam  
- **True Negatives (TN):** Ham messages correctly identified as ham  
- **False Positives (FP):** Ham messages incorrectly labeled as spam  
- **False Negatives (FN):** Spam messages incorrectly labeled as ham  

---

## 📊 Insights and Observations

- Spam messages frequently contain promotional words like *“win”, “free”, “offer”, “click”*.  
- Ham messages are typically shorter and more conversational.  
- False negatives (missed spam) are more critical in real-world scenarios, as they can lead to phishing or scams.  
- Using a larger, more diverse dataset significantly improves real-world performance.
![most common word]()
![top 30 words]()


---

## 🚀 Final Results

- **Best Model:** Multinomial Naive Bayes  
- **Accuracy:** 95.55%  
- **Precision:** 100%  
- **Frameworks Used:** Scikit-learn, NLTK, Pandas  
- **Development Environment:** Google Colab  

The trained model can effectively classify unseen emails as *spam* or *ham* with high reliability.

---

## 🌐 Future Improvements

- Train on even larger and multilingual datasets (real-world email data).  
- Implement deep learning models (LSTM / BERT) for improved contextual understanding.  
- Build a **web interface or API** for real-time spam detection.  
- Enhance preprocessing to detect obfuscated or disguised spam words.

---




---

## 🏁 Conclusion

This project successfully demonstrates how **Natural Language Processing (NLP)** combined with **Machine Learning** can be used to detect and filter spam emails efficiently.  
By leveraging the Naive Bayes algorithm and text vectorization techniques, the system achieves high accuracy and serves as a solid foundation for real-world email filtering solutions.

---

