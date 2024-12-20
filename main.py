import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key from environment variable
# Set OpenAI API key from Streamlit secrets
openai.api_key = st.secrets["openai"]["OPENAI_API_KEY"]

# Default question and answer
DEFAULT_QUESTION = "What is quantum mechanics?"
DEFAULT_ANSWER = (
    "Quantum mechanics is a fundamental theory that describes the behaviour of nature at and below the scale of atoms. "
    "It is the foundation of all quantum physics which includes quantum chemistry, quantum field theory, quantum technology, "
    "and quantum information science."
)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in pdf_document:
        text += page.get_text()
    pdf_document.close()
    return text.strip()

# Function to extract text from TXT file
def extract_text_from_txt(txt_file):
    return txt_file.read().decode("utf-8").strip()

# Function to call the OpenAI API for evaluation
def evaluate_with_openai(question, answer):
    prompt = (
        "You are a sophisticated AI evaluator designed to objectively assess and review responses to various types of assessment questions. "
        "Assume the role of an expert evaluator responsible for scoring test-takers based on their responses.\n\n"
        "### Task\n"
        "Evaluate the response to the following question thoroughly and provide a detailed review using the rubric.\n\n"
        f"### Question\n\"{question}\"\n\n"
        f"### Answer\n\"{answer}\"\n\n"
        "### Evaluation Guidelines\n"
        "- **Validity (4 Marks)**: How well does the answer align with the intent of the question?\n"
        "- **Reliability (3 Marks)**: Is the answer consistent and logically structured?\n"
        "- **Fairness (3 Marks)**: Does the answer fairly address the question without bias?\n"
        "- **Practicality and Usability (3 Marks)**: Is the answer clear and easy to understand?\n"
        "- **Depth and Breadth of Coverage (4 Marks)**: Does the answer comprehensively address core and related aspects?\n"
        "- **Engagement and Originality (3 Marks)**: Is the answer engaging and original?\n"
        "- **Transparency and Clarity (3 Marks)**: Does the answer present ideas clearly and concisely?\n\n"
        "### Output\n"
        "- Parameter-wise scores with justification.\n"
        "- Final score.\n"
        "- Actionable feedback."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # You can change this to other models like "gpt-3.5-turbo"
            messages=[
                {"role": "system", "content": "You are an expert evaluator providing detailed assessments."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"API Request Failed: {str(e)}"

# Streamlit app
def main():
    st.title("Yaksha Ai Auto Evaluation")
    st.write(
        "Enter a question and an answer below, or upload a PDF/TXT file containing the answer. "
        "The app will evaluate the answer using AI."
    )

    # Check if API key is available
    if not openai.api_key:
        st.error("OpenAI API key not found in environment variables. Please add OPENAI_API_KEY to your .env file.")
        return

    # Input fields
    question = st.text_input("Enter Question", value=DEFAULT_QUESTION)
    answer = st.text_area("Enter Answer", value=DEFAULT_ANSWER, height=150)

    # File upload option
    uploaded_file = st.file_uploader("Upload a PDF or TXT file (optional)", type=["pdf", "txt"])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            answer = extract_text_from_pdf(uploaded_file)
            st.info("Extracted text from PDF.")
        elif uploaded_file.type == "text/plain":
            answer = extract_text_from_txt(uploaded_file)
            st.info("Extracted text from TXT file.")

    # Display extracted answer
    st.subheader("Answer Preview")
    st.write(answer)

    # Submit button
    if st.button("Evaluate Answer"):
        st.info("Evaluating answer...")
        evaluation = evaluate_with_openai(question, answer)
        st.subheader("Evaluation Result")
        st.write(evaluation)

if __name__ == "__main__":
    main()
