import streamlit as st
import pickle
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download nltk resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load saved model and vectorizer
model = pickle.load(open("../models/logistic_regression_model.pkl", "rb"))

tfidf = pickle.load(open("../models/tfidf_vectorizer.pkl", "rb"))

# NLP setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):

    text = re.sub(r'<[^>]+>', ' ', text)

    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    text = text.lower()

    tokens = word_tokenize(text)

    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return ' '.join(tokens)

# Streamlit UI
st.title("Amazon Review Sentiment Analysis")

st.write("Predict whether a review is Positive, Neutral, or Negative")

review = st.text_area("Enter Review")

if st.button("Predict Sentiment"):

    cleaned = preprocess_text(review)

    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)[0]

    sentiment_map = {
        0: "Negative 😞",
        1: "Neutral 😐",
        2: "Positive 😊"
    }

    if prediction == 2:
        st.success(f"Prediction: {sentiment_map[prediction]}")

    elif prediction == 1:
        st.warning(f"Prediction: {sentiment_map[prediction]}")

    else:
        st.error(f"Prediction: {sentiment_map[prediction]}")