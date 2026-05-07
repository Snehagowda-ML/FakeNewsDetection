import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# App UI
st.title("📰 Fake News Detection App")
st.subheader("Enter a news article to check if it is Fake or Real")

news_input = st.text_area("Paste your news article here:", height=200)

if st.button("🔍 Analyze News"):
    if news_input.strip() == "":
        st.warning("⚠️ Please enter a news article!")
    else:
        cleaned = clean_text(news_input)
        vectorized = tfidf.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]

        real_prob = round(probability[0] * 100, 2)
        fake_prob = round(probability[1] * 100, 2)

        st.markdown("---")
        if prediction == 1:
            st.error(f"🔴 FAKE NEWS")
        else:
            st.success(f"🟢 REAL NEWS")

        st.markdown("### Prediction Confidence")
        st.progress(int(real_prob))
        col1, col2 = st.columns(2)
        col1.metric("Real News Probability", f"{real_prob}%")
        col2.metric("Fake News Probability", f"{fake_prob}%")