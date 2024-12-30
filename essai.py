import ollama
import streamlit as st
import pdfplumber  
import io
import re
from deep_translator import GoogleTranslator
from langdetect import detect

# Specify the desired model
desiredModel = 'llama3:8b'


def extract_text_and_metadata(pdf_file):
    """Extract text and metadata from a PDF file."""
    pdf_content = []
    try:
        pdf_file.seek(0)  # Reset the file pointer
        with pdfplumber.open(io.BytesIO(pdf_file.read())) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                page_data = {
                    'page_number': page_num + 1,
                    'text': text,
                }
                pdf_content.append(page_data)
    except Exception as e:
        st.error(f"Error extracting PDF content: {str(e)}")
    return pdf_content


def split_text_into_chunks(text, max_words=300):
    """Split text into smaller chunks with a maximum word count."""
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
    """Get Q&A pairs from the NVIDIA chat model."""
    prompt = f"{context}\nProvide 5 relevant questions from this context as well as their answers only based on this context."
    try:
        response = ollama.chat(model=desiredModel, messages=[
            {'role': 'user', 'content': prompt},
        ])
        return response.get('message', {}).get('content', 'No response content')
    except Exception as e:
        return f"Error: {str(e)}"


def detect_language(text):
    """Detect the language of the provided text."""
    try:
        detected_lang = detect(text)
        return detected_lang
    except Exception as e:
        return "en"  # Default to English if detection fails


def translate_text(text, source_lang, target_lang):
    """Translate text from the source language to the target language."""
    try:
        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        return translated
    except Exception as e:
        return f"Translation error: {str(e)}"


def main():
    st.title("PDF to Q&A Converter")
    st.write("Upload a PDF file to extract and format Q&A pairs.")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file:
        # Extract text from the uploaded PDF
        pdf_content = extract_text_and_metadata(uploaded_file)
        

        st.write("Processing uploaded PDF...")

        # Combine all text into a single string
        all_text = " ".join([page['text'] for page in pdf_content if page['text']])
        detected_lang = detect_language(all_text)
        st.write(f"Detected language: {detected_lang}")

        # Split the extracted text into manageable chunks
        chunks = split_text_into_chunks(all_text)
        st.write(f"Total chunks of text: {len(chunks)}")

        # Initialize an output list to hold formatted Q&A pairs
        formatted_output = []

        # Process each chunk
        for i, chunk in enumerate(chunks):
            st.subheader(f"Chunk {i + 1}")

            # Get Q&A from the model for the chunk
            st.write("Generating Q&A...")
            # qa_response = get_qa_from_nvidia_chat(chunk)

            # Translate the Q&A response if necessary
            # if detected_lang != "en":
            qa_response = translate_text(get_qa_from_nvidia_chat(chunk), "en", detected_lang)

            st.text_area(f"Translated Q&A - Chunk {i + 1}", qa_response, height=200)

            # Extract Q&A pairs using regex
            qa_pairs = re.findall(r'\*\*Q\d+:\*\*\s(.*?)\n\*\*A\d+:\*\*\s(.*?)(?:\n|$)', qa_response, re.DOTALL)

            # Format and append each pair to the output list
            for question, answer in qa_pairs:
                formatted_output.append(f"Question: {question}\tAnswer: {answer}")

        output_text = "\n".join(formatted_output)
        output_file = io.StringIO(output_text)
        st.download_button(
            label="Download Q&A File",
            data=output_file.getvalue(),
            file_name="qa_output.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
