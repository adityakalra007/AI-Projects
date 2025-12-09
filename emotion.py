# emotion_app.py
import streamlit as st
from transformers import pipeline
import pandas as pd

# ---------- Configuration ----------
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
# (This model is a compact DistilRoBERTa trained for emotion classification on English text.)

EMOJI_MAP = {
    "anger": "😡",
    "disgust": "🤢",
    "fear": "😨",
    "joy": "😊",
    "neutral": "😐",
    "sadness": "😢",
    "surprise": "😲"
}
# -----------------------------------

st.set_page_config(page_title="AI Emotion Detector", page_icon="🧠😊", layout="wide")

st.title("🧠 AI Emotion Detector")
st.markdown(
    "Detect the emotion expressed in a piece of text. "
    "Enter a sentence or paragraph and the model will predict which emotion is strongest and show confidence scores."
)

with st.expander("How it works (short)"):
    st.write(
        "This app uses a pre-trained Hugging Face Transformer model (DistilRoBERTa) fine-tuned for emotion classification. "
        "It returns a score for each emotion and highlights the top prediction."
    )

# Sidebar / controls
st.sidebar.header("Settings")
st.sidebar.write("Model:")
st.sidebar.write(MODEL_NAME)
show_examples = st.sidebar.checkbox("Show example inputs", value=True)
confidence_threshold = st.sidebar.slider("Minimum confidence to label as confident", 0.0, 1.0, 0.35)

# Load the classification pipeline once and cache it
@st.cache_resource
def load_emotion_pipeline():
    return pipeline("text-classification", model=MODEL_NAME, return_all_scores=True)

pipe = load_emotion_pipeline()

# Input area
col1, col2 = st.columns([3, 1])
with col1:
    text_input = st.text_area("Enter text to analyze:", height=200, placeholder="Type something like: 'I got the job! I'm so happy!'")

    uploaded_file = st.file_uploader("Or upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        try:
            raw = uploaded_file.read().decode("utf-8")
            text_input = raw
            st.info("Loaded text from file.")
        except Exception:
            st.error("Could not read uploaded file (ensure it's UTF-8 .txt).")

    if st.button("Analyze Emotion"):
        if not text_input or text_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                # pipeline returns a list with one element (list of labels+scores) because return_all_scores=True
                results = pipe(text_input)
                # results example: [[{'label':'joy','score':0.9}, ...]]
                scores_list = results[0]
                # convert to dict label->score
                scores = {item['label'].lower(): float(item['score']) for item in scores_list}
                # Ensure we include all known labels (some models may use slightly different label casing)
                df = pd.DataFrame(list(scores.items()), columns=["label", "score"])
                df = df.sort_values("score", ascending=False).reset_index(drop=True)

                top_label = df.loc[0, "label"]
                top_score = df.loc[0, "score"]

                emoji = EMOJI_MAP.get(top_label, "")
                confident = top_score >= confidence_threshold

                # Display results
                st.subheader("Result")
                col_a, col_b = st.columns([2, 3])
                with col_a:
                    st.markdown(f"### {emoji} **{top_label.upper()}**")
                    st.metric("Confidence", f"{top_score:.2%}")
                    if confident:
                        st.success("Prediction is confident ✅")
                    else:
                        st.info("Prediction confidence is low — interpret carefully ℹ️")

                with col_b:
                    # Bar chart of scores (labels on y)
                    chart_data = df.set_index("label")
                    st.bar_chart(chart_data)

                # Show full score table
                with st.expander("Show full scores"):
                    st.table(df.style.format({"score": "{:.3f}"}))

with col2:
    st.markdown("### Tips & Examples")
    if show_examples:
        examples = [
            "I am thrilled — today was the best day of my life!",
            "I can't believe this happened... I'm shaking.",
            "Why would they do that? I'm so angry.",
            "That's disgusting, I don't want to touch it.",
            "Hmm, interesting... not sure what to feel.",
            "I am so sad and overwhelmed.",
            "Wow! That surprised me!"
        ]
        for ex in examples:
            if st.button(f"Use: {ex[:40]}...", key=ex):
                # Place example into the main input box by re-running with JS is tricky; show instruction instead
                st.write("Copy this example and paste it into the text box to analyze.")
                st.code(ex)
    else:
        st.write("Enable example inputs in the sidebar to see sample sentences.")

st.write("---")
st.markdown("**Notes:** The model may not be perfect — for critical uses you should evaluate on your own data. This project is for learning/demo purposes.")

# Footer: credits
st.markdown("Built with ❤️ using Streamlit and Hugging Face Transformers.")
