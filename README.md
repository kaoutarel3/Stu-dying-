# STU(DYING) 
---
This project leverages Streamlit to develop an interactive platform where users can upload PDF documents. The system processes the PDF to automatically generate flashcards, aiding in the extraction and memorization of key information. Additionally, a chatbot is integrated to provide on-demand, context-driven responses based on the content of the uploaded PDF, facilitating efficient information retrieval and enhancing the user learning experience.😊

---
# Building the chatbot
## Outline
The chatbot is designed to provide instant, context-aware responses based on the content of uploaded PDFs.
Here's a simple demo showcasing how the chatbot was built, highlighting the steps , model integration, and creating the user interface.

## Prerequisites
### Python Environment
Python 3.9 or newer.
### Required Python Libraries
.. code-block:: bash

   pip install streamlit pdfplumber PyPDF2 langdetect langchain-community langchain-openai chromadb



# PDF to Q&A Converter
It processes PDF documents to generate **question-and-answer pairs** based on the content. It uses **PyMuPDF** and **pdfplumber** for text and metadata extraction, and leverages NVIDIA's **NeMo Inference API** for generating Q&A responses.

## Features
- Extracts **metadata** and **text** from uploaded PDFs.
- Splits text into manageable chunks for processing.
- Generates **5 relevant questions and answers** from each text chunk using the NVIDIA NeMo API.
- Outputs the Q&A pairs as a downloadable text file.

## Installation

### Prerequisites
1. Python 3.8 or higher.
2. NVIDIA NeMo account with an API key and endpoint.
3. NVIDIA-compatible environment for accessing the NeMo model via the OpenAI client.

### Steps
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Replace placeholders for the API key and endpoint in the code:
   ```python
   API_KEY = '<your API key>'
   ENDPOINT = '<your inference endpoint>'
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the Streamlit app in your browser (usually at `http://localhost:8501`).
2. Upload a PDF file.
3. Wait for the app to process the PDF and generate Q&A pairs.
4. Download the generated Q&A file as a `.txt` file.

## File Structure
- `app.py`: Main application script.
- `requirements.txt`: List of dependencies.
- `README.md`: Documentation for the project.

## Dependencies
- `streamlit`: For the user interface.
- `PyMuPDF (fitz)`: For extracting text and metadata from PDFs.
- `pdfplumber`: For detailed text and table extraction.
- `pandas`: For saving Q&A data (optional in this project).
- `openai`: For interacting with NVIDIA NeMo models.
- `regex`: For parsing Q&A responses.

## How It Works
1. **PDF Processing**:
   - PyMuPDF extracts metadata and text from the PDF.
   - pdfplumber extracts detailed text content.
2. **Text Chunking**:
   - The text is split into chunks (max 300 words) for processing.
3. **Q&A Generation**:
   - Each chunk is sent to the NVIDIA NeMo API to generate relevant Q&A pairs.
4. **Output**:
   - Extracted Q&A pairs are processed and formatted into a downloadable text file.

## Example Output
```
Question: What is the purpose of the Streamlit app?
Answer: The app processes PDF files to extract Q&A pairs.

Question: How are the PDFs processed?
Answer: PDFs are processed using PyMuPDF and pdfplumber for text and metadata extraction.
...
```
