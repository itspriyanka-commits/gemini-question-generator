"""
Multi-Type Question Generator Assistant
-----------------------------------------
Generates different types of questions (MCQ, True/False, Short Answer,
Fill-in-the-Blank) from a given piece of content, using the Google Gemini
API (gemini-flash-latest) with type-specific system instructions.

Run locally:
    streamlit run streamlit_app.py

Deploy:
    Push this repo to GitHub and deploy on Streamlit Community Cloud.
"""

import streamlit as st
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 1. Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Multi-Type Question Generator", page_icon="❓")
st.title("❓ Multi-Type Question Generator Assistant")
st.caption("Powered by Google Gemini (gemini-flash-latest)")

# ---------------------------------------------------------------------------
# 2. API key handling
#    - On Streamlit Community Cloud: add GEMINI_API_KEY to Secrets.
#    - Locally: falls back to a sidebar text input.
# ---------------------------------------------------------------------------
api_key = st.secrets.get("GEMINI_API_KEY", None) if hasattr(st, "secrets") else None

if not api_key:
    api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")

if not api_key:
    st.warning("Please provide a Gemini API key in the sidebar (or in Secrets) to continue.")
    st.stop()

# Initialize the Gemini client using the loaded API key
client = genai.Client(api_key=api_key)

# ---------------------------------------------------------------------------
# 3. Question type -> system instruction mapping
# ---------------------------------------------------------------------------
question_types = {
    "MCQ": "You are a quiz question generator. Given the content, create "
           "multiple-choice questions with 4 options each (labeled A-D) "
           "and clearly indicate the correct answer. Respond with ONLY "
           "the questions, no extra commentary.",
    "True/False": "You are a quiz question generator. Given the content, "
                   "create True/False statements based on it, and clearly "
                   "indicate whether each statement is True or False. "
                   "Respond with ONLY the questions, no extra commentary.",
    "Short Answer": "You are a quiz question generator. Given the content, "
                     "create short-answer questions that can be answered "
                     "in one or two sentences, along with a brief model "
                     "answer for each. Respond with ONLY the questions, "
                     "no extra commentary.",
    "Fill in the Blank": "You are a quiz question generator. Given the "
                          "content, create fill-in-the-blank questions by "
                          "removing a key term or phrase from a sentence "
                          "and replacing it with a blank ____, and provide "
                          "the correct answer for each. Respond with ONLY "
                          "the questions, no extra commentary.",
}

# ---------------------------------------------------------------------------
# 4. Core question generation function
# ---------------------------------------------------------------------------
def question_generator(content: str, question_type: str, num_questions: int = 5) -> str:
    """
    Generate `num_questions` questions of `question_type` from `content`
    using Gemini.

    Args:
        content: The source text to generate questions from.
        question_type: One of the keys in the `question_types` dict.
        num_questions: How many questions to generate.

    Returns:
        The generated questions (string only).
    """
    system_instruction = question_types[question_type]
    prompt = f"Generate {num_questions} questions from this content:\n\n{content}"

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction
        ),
    )

    return response.text


# ---------------------------------------------------------------------------
# 5. Streamlit UI
# ---------------------------------------------------------------------------
content = st.text_area(
    "Paste your content here:",
    placeholder="e.g. Paste a paragraph, article, or study notes...",
    height=200,
)
col1, col2 = st.columns(2)
with col1:
    question_type = st.selectbox("Select question type:", list(question_types.keys()))
with col2:
    num_questions = st.number_input("Number of questions:", min_value=1, max_value=20, value=5)

if st.button("Generate Questions", type="primary"):
    if not content.strip():
        st.error("Please paste some content first.")
    else:
        with st.spinner(f"Generating {question_type} questions..."):
            try:
                result = question_generator(content, question_type, num_questions)
                st.success("Generated Questions:")
                st.write(result)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
