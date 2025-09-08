import streamlit as st
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

import pandas as pd
import re
import wikipediaapi
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Language detection + translation
from langdetect import detect
from googletrans import Translator

translator = Translator()

# -------------------------
# 1. Load & Train Models
# -------------------------
@st.cache_resource
def load_models():
    data = pd.read_csv("news.csv")   # Your dataset with 'text' & 'label'

    # Convert labels to numeric
    data["label"] = data["label"].map({"REAL": 1, "FAKE": 0})

    # Clean text (allow unicode letters)
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)  # keep all words
        return text

    data["text"] = data["text"].apply(clean_text)

    # Split
    x = data["text"]
    y = data["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=5000)
    xv_train = vectorizer.fit_transform(x_train)

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Passive Aggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=42),
    }

    trained_models = {}
    for name, clf in models.items():
        clf.fit(xv_train, y_train)
        trained_models[name] = clf

    return trained_models, vectorizer


models, vectorizer = load_models()

# -------------------------
# 2. Streamlit UI
# -------------------------

st.title("📰 Fake News & Fact Checker (Multi-Language)")
st.markdown("### Paste any news article or claim to check if it's **Fake or Real** using ML models, or verify facts with **Wikipedia**.")

# User input
news_input = st.text_area("✍️ Enter News Text Here:", height=150, placeholder="Type or paste news text...")

# -------------------------
# 3. Helper Functions
# -------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return text


def detect_and_translate(text):
    try:
        detected_lang = detect(text)
    except:
        detected_lang = "en"

    if detected_lang != "en":
        translated = translator.translate(text, src=detected_lang, dest="en").text
    else:
        translated = text

    return detected_lang, translated


# -------------------------
# 4. Fake News Classification
# -------------------------

col1, col2 = st.columns(2)

with col1:
    if st.button("🚀 Check Fake/Real"):
        if news_input.strip() == "":
            st.warning("⚠️ Please enter some news text first.")
        elif len(news_input.split()) < 10:
            st.warning("⚠️ Text is too short to classify reliably. Please enter a longer article or headline.")
        else:
            detected_lang, translated_text = detect_and_translate(news_input)

            cleaned = clean_text(translated_text)
            vectorized = vectorizer.transform([cleaned])

            results = {}
            for name, clf in models.items():
                pred = clf.predict(vectorized)[0]
                results[name] = "✅ Real" if pred == 1 else "❌ Fake"

            # Show results
            with st.expander("🔍 Model Predictions"):
                for model_name, prediction in results.items():
                    st.write(f"**{model_name}:** {prediction}")

            # Majority voting
            votes = [1 if v == "✅ Real" else 0 for v in results.values()]
            final = "✅ Real" if sum(votes) >= 2 else "❌ Fake"

            st.subheader("🗳️ Final Verdict")
            if final == "✅ Real":
                st.success(f"This news looks **Real** ✅ (Language detected: {detected_lang.upper()})")
            else:
                st.error(f"This news looks **Fake** ❌ (Language detected: {detected_lang.upper()})")


# -------------------------
# 5. Enhanced Fact Checking via Wikipedia
# -------------------------

with col2:
    if st.button("🔎 Fact Check (Wikipedia)"):
        if news_input.strip() == "":
            st.warning("⚠️ Please enter some text first.")
        else:
            with st.spinner("🔎 Checking Wikipedia..."):
                try:
                    detected_lang, translated_text = detect_and_translate(news_input)

                    # Use Wikipedia in detected language
                    wiki = wikipediaapi.Wikipedia(
                        language=detected_lang,
                        user_agent="FakeNewsDetectorApp/1.0 (contact: your-email@example.com)"
                    )

                    # Use first few words as subject (NER would require language models)
                    subject = " ".join(news_input.split()[:5])

                    page = wiki.page(subject)

                    if not page.exists():
                        st.error(f"❌ Could not find this topic on Wikipedia ({detected_lang.upper()}).")
                    else:
                        summary = page.summary[:600].lower()
                        input_words = news_input.lower().split()

                        matched = sum([1 for w in input_words if w in summary])
                        similarity = matched / len(input_words) if input_words else 0

                        sensitive_words = ["dead", "death", "died", "murdered", "killed"]
                        if any(word in news_input.lower() for word in sensitive_words):
                            if not any(word in summary for word in sensitive_words):
                                st.error("❌ This claim is FALSE (contradicts Wikipedia)")
                            else:
                                st.success("✅ This claim may be TRUE (Wikipedia confirms)")
                        else:
                            if similarity > 0.35:
                                st.success(f"✅ Likely TRUE (Confidence: High, {similarity:.0%} word match)")
                            elif similarity > 0.20:
                                st.warning(f"⚠️ Unclear (Confidence: Medium, {similarity:.0%} word match)")
                            else:
                                st.error(f"❌ Possibly FALSE (Confidence: Low, {similarity:.0%} word match)")

                        with st.expander("📖 Wikipedia Reference"):
                            st.info(page.summary[:600])

                except Exception as e:
                    st.warning(f"⚠️ Could not verify (Error: {e})")
