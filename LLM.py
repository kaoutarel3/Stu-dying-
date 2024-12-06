import streamlit as st
import os 
import PyPDF2


from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from ingest import initialize_vector_store


# function to read the pdf file
def read_pdf(file):
    pdfReader = PyPDF2.PdfReader(file)
    all_page_text = ""
    for page in pdfReader.pages:
        all_page_text += page.extract_text() + "\n"
    return all_page_text


def retrieve_from_db(question):
    # get the model
    model = ChatOllama(model="llama3.2")
    # initialize the vector store
    db = initialize_vector_store()

    retriever = db.similarity_search(question, k=2)
    after_rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    if there is no answer, please answer with "I m sorry, the context is not enough to answer the question."
    """

    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)

    after_rag_chain = (
        {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
        | after_rag_prompt
        | model
        | StrOutputParser()
    )

    return after_rag_chain.invoke({"context": retriever, "question": question})


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
    







st.title("CHATBOT")
st.write("Welcome to our chatbot")
file = st.file_uploader("Upload a PDF file", type=["pdf"])
if file:
    doc = read_pdf(file)
    question = st.text_input("Ask a question")
    if st.button("Ask"):
        answer = retriever(doc, question)
        st.write(answer)
else:
    question = st.text_input("Ask a question")
    if st.button("Ask"):
        answer = retrieve_from_db(question)
        answer = get_chatbot_response(question)
        st.write(answer)



