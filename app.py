import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE-API_KEY"))

# Define functions (same as in your code)
def get_pdf_text(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed a possible from the provided context, make sure to prompt
    provided context just say, "answer is not available in the context", don't provide the
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """
    model = GoogleGenerativeAI(model="gemini-1.0")  # Make sure to define model here
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain, model  # Return model as well

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    if not docs:
        st.write("No relevant context found in the PDF.")
        return

    chain, model = get_conversational_chain()  # Get model and chain from this function
    # Use the chain to process the response
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    if "output_text" in response:
        st.write(f"**Answer:** {response['output_text']}")
    else:
        st.write("The model could not generate a response.")

# Streamlit UI Design with Custom CSS
st.set_page_config(page_title="Chat with Multiple PDFs using Gemini", page_icon="📄", layout="wide")

# Add Custom CSS using st.markdown
st.markdown("""
    <style>
        /* Custom CSS Styling */
        .stApp {
            background-color: #f4f4f9;
            font-family: 'Arial', sans-serif;
        }

        .css-1v0mbdj {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            text-align: center;
            font-size: 24px;
        }

        .stButton button {
            background-color: #007bff;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            width: 100%;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        .stButton button:hover {
            background-color: #0056b3;
        }

        .stFileUploader {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 24px;
        }

        .stTextInput input {
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 16px;
            width: 100%;
            border: 1px solid #cccccc;
            margin-bottom: 16px;
        }

        .stTextInput input:focus {
            border-color: #007bff;
        }

        .stMarkdown {
            font-size: 18px;
            margin-bottom: 30px;
        }

        .stSuccess {
            background-color: #28a745;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }

        .stError {
            background-color: #dc3545;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }

        .stSpinner {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Main Section
st.title("Chat with Multiple PDFs using Gemini")
st.markdown("""
    ## Welcome to the PDF Chat Application!
    
    Upload a PDF document and ask questions based on the contents. The application uses advanced AI to extract meaning from the document and answer your questions.

    **How to use**:
    1. Upload a PDF file in the sidebar.
    2. Ask a question from the document using the text input box.
    3. The model will provide an answer based on the uploaded content.
""")

# Sidebar Section for file upload and processing
with st.sidebar:
    st.title("Menu")
    st.markdown("### Upload PDF")
    pdf_docs = st.file_uploader("Upload your PDF File", type="pdf")
    if st.button("Submit & Process"):
        if pdf_docs is not None:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing Complete! You can now ask questions.")
        else:
            st.error("Please upload a PDF file first.")

# Question input section
user_question = st.text_input("Ask a question from the uploaded PDF:")

# Display answer
if user_question:
    user_input(user_question)
