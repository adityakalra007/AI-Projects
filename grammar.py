import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Grammar Corrector", page_icon="✔️")

st.title("✔️ AI Grammar & Spelling Corrector")
st.markdown("""
Automatically correct grammar and spelling mistakes in your text using **AI**.  
Just type your sentence or paragraph and click **Correct**.
""")

# Input text
text_input = st.text_area("Enter text here:", height=200)

# Generate button
if st.button("Correct Text"):
    if not text_input:
        st.warning("Please enter some text!")
    else:
        try:
            # Load grammar correction model
            corrector = pipeline("text2text-generation", model="oliverguhr/spelling-correction-english-base")
            
            with st.spinner("Correcting..."):
                corrected = corrector(text_input, max_length=200)
            
            st.success("Corrected Text:")
            st.write(corrected[0]["generated_text"])
        except Exception as e:
            st.error(f"Error: {e}")
