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
from deep_translator import GoogleTranslator


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

# Create two columns
col1, col2 = st.columns(2)

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
        translated = GoogleTranslator(source=detected_lang, target="en").translate(text)
    else:
        translated = text

    return detected_lang, translated


def extract_years(text):
    """Extract all 4-digit years from text"""
    return set(re.findall(r"\b\d{4}\b", text))


# -------------------------
# 4. Fake News Classification (future expansion)
# -------------------------

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

                    # Initialize Wikipedia API
                    wiki = wikipediaapi.Wikipedia(
                        language=detected_lang,
                        user_agent="FakeNewsDetectorApp/1.0 (contact: your-email@example.com)"
                    )

                    # Use full translated text as subject
                    subject = translated_text.strip()
                    page = wiki.page(subject)

                    if not page.exists():
                        st.error(f"❌ Could not find this topic on Wikipedia ({detected_lang.upper()}).")
                    else:
                        summary = page.summary[:600].lower()
                        input_words = translated_text.lower().split()

                        # ✅ Step 1: year/date check
                        input_years = extract_years(translated_text)
                        summary_years = extract_years(summary)

                        if input_years and not (input_years & summary_years):
                            st.error("❌ This claim is FALSE (date mismatch with Wikipedia)")
                        else:
                            # ✅ Step 2: similarity check
                            matched = sum([1 for w in input_words if w in summary])
                            similarity = matched / len(input_words) if input_words else 0

                            sensitive_words = ["dead", "death", "died", "murdered", "killed"]
                            if any(word in translated_text.lower() for word in sensitive_words):
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
