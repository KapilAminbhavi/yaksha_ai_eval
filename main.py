import streamlit as st
import fitz  # PyMuPDF for PDF text extraction
import requests  # For making API requests
import os

# Default question and answer
DEFAULT_QUESTION = "What is quantum mechanics?"
DEFAULT_ANSWER = (
    "Quantum mechanics is a fundamental theory that describes the behaviour of nature at and below the scale of atoms. "
    "It is the foundation of all quantum physics which includes quantum chemistry, quantum field theory, quantum technology, "
    "and quantum information science."
)

# LLaMA3 API endpoint
API_URL = "https://103.168.74.48/generate"

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

# Function to call the LLaMA3 API for evaluation
def call_llama3_api(question, answer):
    payload = {
        "prompt": (
            f"You are a sophisticated AI evaluator designed to objectively assess and review responses to various types of assessment questions. "
            f"Assume the role of an expert evaluator responsible for scoring test-takers based on their responses.\n\n"
            f"### Task\n"
            f"Evaluate the response to the following question thoroughly and provide a detailed review using the rubric.\n\n"
            f"### Question\n\"{question}\"\n\n"
            f"### Answer\n\"{answer}\"\n\n"
            f"### Evaluation Guidelines\n"
            f"- **Validity (4 Marks)**: How well does the answer align with the intent of the question?\n"
            f"- **Reliability (3 Marks)**: Is the answer consistent and logically structured?\n"
            f"- **Fairness (3 Marks)**: Does the answer fairly address the question without bias?\n"
            f"- **Practicality and Usability (3 Marks)**: Is the answer clear and easy to understand?\n"
            f"- **Depth and Breadth of Coverage (4 Marks)**: Does the answer comprehensively address core and related aspects?\n"
            f"- **Engagement and Originality (3 Marks)**: Is the answer engaging and original?\n"
            f"- **Transparency and Clarity (3 Marks)**: Does the answer present ideas clearly and concisely?\n\n"
            f"### Output\n"
            f"- Parameter-wise scores with justification.\n"
            f"- Final score.\n"
            f"- Actionable feedback.\n\n"
            f"Now evaluate and score the response."
        ),
        "session_id": "streamlit_app",
        "max_length": 300,
        "question": question,
        "answer": answer
    }

    try:
        response = requests.post(API_URL, json=payload, verify=False)  # Skipping SSL verification for demo
        if response.status_code == 200:
            return response.json().get("response", "No response received from LLaMA3.")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return f"API Request Failed: {str(e)}"

# Streamlit app
def main():
    st.title("LLaMA3 Evaluation App")
    st.write(
        "Enter a question and an answer below, or upload a PDF/TXT file containing the answer. "
        "The app will send the input to the LLaMA3 API and display the evaluation."
    )

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
        st.info("Sending data to LLaMA3 API...")
        evaluation = call_llama3_api(question, answer)
        st.subheader("Evaluation Result")
        st.write(evaluation)

if __name__ == "__main__":
    main()
