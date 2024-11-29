import os
import streamlit as st
import fitz  
import pdfplumber  
import pandas as pd  
import io
from openai import OpenAI  
import re


API_KEY = '<your API key>'  
ENDPOINT = '<your inference endpoint>'  


client = OpenAI(
  base_url=ENDPOINT,
  api_key=API_KEY
)

def extract_text_and_metadata(pdf_file):
    # Open the PDF with PyMuPDF to extract metadata
    pdf = fitz.open(stream=pdf_file.read(), filetype="pdf")
    metadata = pdf.metadata

    # Container for extracted content
    pdf_content = []

    # Open the PDF with pdfplumber for text extraction
    pdf_file.seek(0)  # Reset file pointer for pdfplumber
    with pdfplumber.open(io.BytesIO(pdf_file.read())) as pdf_plumber:
        for page_num, page in enumerate(pdf_plumber.pages):
            # Extract text using pdfplumber
            text = page.extract_text()

            # Add extracted text and page number to the content list
            page_data = {
                'page_number': page_num + 1,
                'text': text,
            }
            pdf_content.append(page_data)

    # Close the PyMuPDF PDF
    pdf.close()

    # Return metadata and extracted content
    return metadata, pdf_content

def split_text_into_chunks(text, max_words=300):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(current_chunk) + 1 > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
def get_qa_from_nvidia_chat(context):
    prompt = f"{context}\nProvide 5 relevant questions from this context as well as their answers only based on this context and in the same language provided."

    try:
        completion = client.chat.completions.create(
            model="mistralai/mixtral-8x7b-instruct-v0.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            top_p=1,
            max_tokens=1024,
            stream=False  # Disable streaming
        )

        # Directly return the full response content
        return completion.choices[0].message.content

    except Exception as e:
        return f"Error: {str(e)}"

    
def process_qa_responses(content):
    """
    Processes the Q&A responses and returns the formatted output as a string (to be used for download).
    """
    output = []  # Using a list to collect the output

    for page in content:
        text = page['text']
        chunks = split_text_into_chunks(text)

        # For each chunk, generate the Q&A pairs and process the response
        for chunk in chunks:
            # Generate the Q&A response from the model
            qa_response = get_qa_from_nvidia_chat(chunk)
            # Use regex to extract Question-Answer pairs
            qa_pairs = re.findall(r'Question:\s(.*?)\nAnswer:\s(.*?)\n', qa_response, re.DOTALL)

            # Format and append each pair to the output
            for question, answer in qa_pairs:
                output.append(f"{question}\t{answer}")

    # Return all Q&A pairs as a single string, joining by newlines
    return "\n".join(output)


def main():
    """
    Streamlit app for uploading a PDF file and downloading the processed Q&A txt file.
    """
    st.title("PDF to Q&A Converter")
    st.write("Upload a PDF file to extract and format Q&A pairs.")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file:
        # Extract text from the uploaded PDF
        metadata, content = extract_text_and_metadata(uploaded_file)
        
        # Process the content and generate Q&A pairs
        formatted_output = process_qa_responses(content)
        
        # Convert the formatted output into a StringIO object (in-memory text file)
        output_file = io.StringIO(formatted_output)

        # Allow user to download the resulting Q&A txt file
        st.download_button(
            label="Download Q&A File",
            data=output_file.getvalue(),  # Get the content of the in-memory file
            file_name="qa_output.txt",
            mime="text/plain",
        )

if __name__ == "__main__":
    main()