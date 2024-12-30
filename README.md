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
```bash
    pip install streamlit pdfplumber PyPDF2 langdetect langchain-community langchain-openai chromadb
```

### Ollama API
1. Download Ollama
2. Download Required Models
   ```bash
   ollama pull llama3.2
   ollama pull mxbai-embed-large:latest

### Importing libraries
```python
    import streamlit as st
    import os 
    import PyPDF2
    import ollama
    import pdfplumber
    from langdetect import detect
    from langchain_community.vectorstores import Chroma
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings.ollama import OllamaEmbeddings
    from langchain_openai import ChatOpenAI
    from langchain.schema import Document
    from langchain.text_splitter import CharacterTextSplitter
```

### Reading the PDF file
```python
def read_pdf(file):
    pdfReader = PyPDF2.PdfReader(file)
    all_page_text = ""
    for page in pdfReader.pages:
        all_page_text += page.extract_text() + "\n"
    return all_page_text
```

### Retrieve and respond to queries from a PDF file
```python
def retriever(doc, question):
    model_local = ChatOllama(model="llama3.2")

    doc = Document(page_content=doc)
    doc = [doc]
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=800, chunk_overlap=0)
    doc_splits = text_splitter.split_documents(doc)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OllamaEmbeddings(model="mxbai-embed-large:latest"),
    )
    retriever = vectorstore.as_retriever(k=2)
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    if there is no answer, please answer with "I m sorry, the context is not enough to answer the question."
    """
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model_local
        | StrOutputParser()
    )

    return after_rag_chain.invoke(question)
```

The model used in the retriever function is ChatOllama, specifically the **"llama3.2"** variant.
It is a conversational AI model,introduced by Meta in 2024, designed to process and generate human-like responses based on input prompts.
In this function, it serves as the final step in generating answers to questions, leveraging the context retrieved from a vector store.

**Additional Models Embedded:**

**"mxbai-embed-large"** is an embedding model used to convert the text from the documents into numerical vectors for similarity comparisons

### Generate Chatbot response with language detection
```python
def get_chatbot_response(prompt):
    
    detected_language = detect(prompt)
    
    # Modify the prompt based on detected language
    if detected_language == 'fr':
        modified_prompt = "Veuillez répondre en français: " + prompt
    else:
        modified_prompt = prompt  # Default to English if language is not French
    
    # Call Ollama's API with the modified prompt
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": modified_prompt}])
    
    # Assuming the response is a dictionary-like object, get the text from the response
    return response['text']
```

# Flashcads generation
## Outline

The PDF to Q&A Converter is a Streamlit application designed to extract text from uploaded PDF files and generate context-aware Q&A pairs using an advanced AI model. This tool supports multi-language PDFs, automated chunking of text, and language translation for comprehensive usability.

## Prerequisites
###Python Environment
Python 3.9 or newer.
### Required Python Libraries
```bash
 pip install streamlit pdfplumber langdetect deep-translator
```
### Ollama API
1. Download Ollama
2. Download Required Models
   ```bash
    ollama pull llama3.2
   ```
   ## Importing libraries
   ```python
    import streamlit as st
    import pdfplumber
    import io
    import re
    from deep_translator import GoogleTranslator
    from langdetect import detect
   ```
   ## Reading the PDF File
   ```python
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

## Splitting Text into Chunks
------------------------------
```python

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
```
## Detecting and Translating Language
```python

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
```
The application integrates AI-driven Q&A generation with language detection and translation, ensuring accessibility for users worldwide. The **"llama3:8b"** model is employed for contextual understanding and response generation.




        


   
