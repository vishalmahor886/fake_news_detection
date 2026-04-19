import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Load resources
# -----------------------------
@st.cache_resource
def load_all():
    model = load_model("model.h5")

    with open("config.pkl", "rb") as f:
        config = pickle.load(f)

    return model, config

model, config = load_all()

vec_size = config["vec_size"]
sent_length = config["sent_length"]

# -----------------------------
# NLTK setup
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# -----------------------------
# Text preprocessing
# -----------------------------
def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)

    y = []
    for i in words:
        if i.isalnum() and i not in stop_words:
            y.append(ps.stem(i))

    return ' '.join(y)

# -----------------------------
# Prediction function
# -----------------------------
def predict_news(text):
    text = transform_text(text)

    onehot = one_hot(text, vec_size)
    padded = pad_sequences([onehot], maxlen=sent_length, padding='post')

    pred = model.predict(padded)

    return pred[0][0]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection App")
st.write("Enter a news article or headline to check if it's Fake or Real.")

user_input = st.text_area("✍️ Enter News Text", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        score = predict_news(user_input)

        if score < 0.5:
            st.error(f"🚨 Fake News (Confidence: {1-score:.2f})")
        else:
            st.success(f"✅ Real News (Confidence: {score:.2f})")