import streamlit as st
import pandas as pd
import re
import wikipediaapi
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# 1. Load & Train Models
# -------------------------
@st.cache_resource
def load_models():
    data = pd.read_csv("news.csv")   # Your dataset with 'text' & 'label'

    # Convert labels to numeric
    data["label"] = data["label"].map({"REAL": 1, "FAKE": 0})

    # Clean text
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
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
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 Fake News & Fact Checker")
st.markdown("### Paste any news article or claim to check if it's **Fake or Real** using ML models, or verify facts with **Wikipedia**.")

# User input
news_input = st.text_area("✍️ Enter News Text Here:", height=150, placeholder="Type or paste news text...")

# Layout for buttons
col1, col2 = st.columns(2)

# -------------------------
# 3. Fake News Classification
# -------------------------
with col1:
    if st.button("🚀 Check Fake/Real"):
        if news_input.strip() == "":
            st.warning("⚠️ Please enter some news text first.")
        elif len(news_input.split()) < 10:
            st.warning("⚠️ Text is too short to classify reliably. Please enter a longer article or headline.")
        else:
            def clean_text(text):
                text = str(text).lower()
                text = re.sub(r"[^a-zA-Z\s]", "", text)
                return text

            cleaned = clean_text(news_input)
            vectorized = vectorizer.transform([cleaned])

            results = {}
            for name, clf in models.items():
                pred = clf.predict(vectorized)[0]
                results[name] = "✅ Real" if pred == 1 else "❌ Fake"

            # Show results in expander
            with st.expander("🔍 Model Predictions"):
                for model_name, prediction in results.items():
                    st.write(f"**{model_name}:** {prediction}")

            # Majority Voting
            votes = [1 if v == "✅ Real" else 0 for v in results.values()]
            final = "✅ Real" if sum(votes) >= 2 else "❌ Fake"

            st.subheader("🗳️ Final Verdict")
            if final == "✅ Real":
                st.success("This news looks **Real** ✅")
            else:
                st.error("This news looks **Fake** ❌")

# -------------------------
# 4. Enhanced Fact Checking via Wikipedia
# -------------------------
import spacy
from spacy.cli import download

# Download the English model if not present
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")



import spacy
import wikipediaapi

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

with col2:
    if st.button("🔎 Fact Check (Wikipedia)"):
        if news_input.strip() == "":
            st.warning("⚠️ Please enter some text first.")
        else:
            try:
                # -------------------------
                # 1. Extract subject using NER
                # -------------------------
                doc = nlp(news_input)
                entities = [ent.text for ent in doc.ents]

                if entities:
                    subject = entities[0]  # pick first named entity
                else:
                    # fallback: first 3 words capitalized
                    subject = " ".join([w.capitalize() for w in news_input.split()[:3]])

                # -------------------------
                # 2. Wikipedia Query
                # -------------------------
                wiki = wikipediaapi.Wikipedia(
                    language="en",
                    user_agent="FakeNewsDetectorApp/1.0 (contact: your-email@example.com)"
                )
                page = wiki.page(subject)

                if not page.exists():
                    st.error("❌ Could not find this topic on Wikipedia.")
                else:
                    summary = page.summary[:600].lower()  # first 600 chars
                    input_words = news_input.lower().split()

                    # -------------------------
                    # 3. Word-overlap similarity
                    # -------------------------
                    matched = sum([1 for w in input_words if w in summary])
                    similarity = matched / len(input_words) if input_words else 0

                    # -------------------------
                    # 4. Sensitive word check
                    # -------------------------
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
