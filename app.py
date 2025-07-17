import streamlit as st
from PIL import Image
import openai
import os
import base64
from dotenv import load_dotenv

# Load API key
from pathlib import Path

dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path)
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="DIY Advisor Bot", layout="wide")

st.title("🔧 DIY Advisor Bot")
st.subheader("Snap a photo, speak your question, get help at your level.")

# Inputs
tools = st.text_input("What tools do you have?")
experience = st.selectbox("Your experience level:", ["Beginner", "Intermediate", "Advanced"])
uploaded_image = st.file_uploader("Upload a photo of your project", type=["jpg", "jpeg", "png"])
question_text = st.text_input("Or type your question here:")

# GPT Vision call function
def analyze_image_with_prompt(image_bytes, tools, experience, question):
    prompt = f"""
You are a DIY project assistant helping a user with a task. Here's their profile:
- Tools: {tools}
- Experience: {experience}
- Question: {question}

Look at the uploaded image and describe what you're seeing. Then, based on the user's skill level and tools, offer:
1. A list of tools required (from their existing tools if possible)
2. Step-by-step instructions
3. Any warnings or common mistakes
4. Optional: a relevant YouTube search query

Use clear, structured formatting. Keep it friendly and practical.
"""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_bytes}}
            ]}
        ],
        max_tokens=1000
    )

    return response.choices[0].message.content


# When user clicks the button
if st.button("Get Advice"):
    if uploaded_image and question_text:
        bytes_data = uploaded_image.read()
        image_base64 = base64.b64encode(bytes_data).decode("utf-8")

        with st.spinner("Analyzing your image..."):
            result = analyze_image_with_prompt(
                image_base64,
                tools,
                experience,
                question_text
            )
            st.markdown(result)
    else:
        st.warning("Please upload an image and enter a question.")
