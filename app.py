import wikipediaapi

# -------------------------
# 4. Enhanced Fact Checking via Wikipedia
# -------------------------
with col2:
    if st.button("🔎 Fact Check (Wikipedia)"):
        if news_input.strip() == "":
            st.warning("⚠️ Please enter some text first.")
        else:
            try:
                # Use first 3 words as subject guess
                subject = " ".join(news_input.split()[:3])

                wiki = wikipediaapi.Wikipedia("en")
                page = wiki.page(subject)

                if not page.exists():
                    st.error("❌ Could not find this topic on Wikipedia.")
                else:
                    summary = page.summary[:600].lower()  # first 600 chars
                    words = news_input.lower().split()

                    # Word overlap similarity
                    matched = sum([1 for w in words if w in summary])
                    similarity = matched / len(words) if words else 0

                    # Sensitive word check
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
                        st.info(summary)

            except Exception as e:
                st.warning(f"⚠️ Could not verify (Error: {e})")
