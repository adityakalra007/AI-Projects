import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load models
st.write("Hello from inside the app!")


@st.cache_resource
def load_summarizer():
    return pipeline("summarization")

@st.cache_resource
def load_paraphraser():
    tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")
    return tokenizer, model

@st.cache_resource
def load_corrector():
    return pipeline("text2text-generation", model="oliverguhr/spelling-correction-english-base")


# UI Settings
st.set_page_config(
    page_title="AI Writing Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.title("✨ AI Writing Tools")
option = st.sidebar.radio(
    "Select Tool",
    ("Text Summarizer", "Text Paraphraser", "Grammar Corrector")
)

st.sidebar.info("Made with ❤️ using Streamlit + Transformers")

# Main Title
st.title("🧠 AI Writing Assistant — Smart NLP Tools")

st.markdown("""
This tool uses state-of-the-art **AI language models** to help with:
- ✨ Text Summarization  
- 🔄 Paraphrasing / Rewriting  
- ✔️ Grammar Correction  
""")

st.write("---")

# Tool: Text Summarizer
if option == "Text Summarizer":
    st.header("✨ AI Text Summarizer")
    text = st.text_area("Enter your long text here:", height=200)

    if st.button("Summarize"):
        if len(text) < 30:
            st.warning("Please enter a longer text.")
        else:
            summarizer = load_summarizer()
            summary = summarizer(text, max_length=80, min_length=30, do_sample=False)
            st.success("Summary Generated:")
            st.write(summary[0]["summary_text"])

# Tool: Paraphraser
elif option == "Text Paraphraser":
    st.header("🔄 AI Text Paraphraser")
    sentence = st.text_area("Enter text to paraphrase:", height=150)

    if st.button("Paraphrase"):
        if len(sentence) < 10:
            st.warning("Please enter a valid sentence.")
        else:
            tokenizer, model = load_paraphraser()
            input_text = "paraphrase: " + sentence
            inputs = tokenizer.encode_plus(input_text, return_tensors="pt", padding=True)
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=60
            )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.success("Paraphrased Text:")
            st.write(result)

# Tool: Grammar Corrector
elif option == "Grammar Corrector":
    st.header("✔️ AI Grammar & Spelling Corrector")
    sentence = st.text_area("Enter text to correct:", height=150)

    if st.button("Correct"):
        corrector = load_corrector()
        output = corrector(sentence, max_length=200)
        st.success("Corrected Sentence:")
        st.write(output[0]["generated_text"])
