import streamlit as st
import openai
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(page_title="AI Image Generator", page_icon="🎨")

st.title("🎨 AI Image Generator")
st.markdown("""
Generate amazing images from text prompts using **OpenAI DALL·E (gpt-image-1)**.  
Enter your API key, type a prompt, and click generate!  
""")

# Sidebar for API key
st.sidebar.header("API Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Prompt input
prompt = st.text_area("Enter your image description here:", height=150)

# Image size selection
size = st.selectbox("Select Image Size", ["256x256", "512x512", "1024x1024", "auto"])

# Generate button
if st.button("Generate Image"):
    if not api_key:
        st.warning("Please enter your API key!")
    elif not prompt:
        st.warning("Please enter a prompt!")
    else:
        try:
            # Set API key
            openai.api_key = api_key

            # Use the new create endpoint
            response = openai.Image.create(
                model="gpt-image-1",
                prompt=prompt,
                size=size,
                n=1
            )

            image_url = response['data'][0]['url']

            # Fetch image
            img_data = requests.get(image_url).content
            image = Image.open(BytesIO(img_data))

            st.image(image, caption="Generated Image", use_column_width=True)
            st.success("Image generated successfully!")

        except Exception as e:
            st.error(f"Error: {e}")
