

# Streamlit imports
import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import os
from PIL import Image
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Set environment variables for API keys (if needed)
os.environ["GROQ_API_KEY"] = "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"

# Initialize LLM and document processing setup (based on CAP)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
embedding = HuggingFaceEmbeddings()

# Streamlit UI
st.title("Super Fantastic Chatbot")
st.subheader("Query the HDFC24 PDF document using natural language")

# Load the pre-existing HDFC24 PDF document (replace with correct file path)
pdf_path = '/mnt/data/hdfc24.pdf'

# Function to extract text from image-based PDFs using Tesseract
def extract_text_from_image_pdf(pdf_path, display_images=False, save_images=False):
    # Convert PDF to a list of images
    images = convert_from_path(pdf_path)
    
    # Initialize an empty string to hold the extracted text
    extracted_text = ""
    
    # Loop through each image and use Tesseract to extract text
    for i, image in enumerate(images):
        # Extract text using Tesseract OCR
        text = pytesseract.image_to_string(image)
        extracted_text += text + "\n"
        
        # Display images in the Streamlit app
        if display_images:
            st.image(image, caption=f'Page {i+1}', use_column_width=True)
        
        # Save images if required
        if save_images:
            image.save(f"page_{i+1}.png", "PNG")
    
    return extracted_text

# Load the PDF document
if os.path.exists(pdf_path):
    # Use UnstructuredPDFLoader for regular PDF text extraction
    loader = UnstructuredPDFLoader(pdf_path)
    documents = loader.load()

    # If the document has no text, assume it might be image-based and use OCR
    if len(documents) == 0:
        st.write("No text found in the PDF, attempting OCR extraction...")

        # Option to display or save the images
        display_images = st.checkbox("Display PDF pages as images?")
        save_images = st.checkbox("Save PDF pages as images?")
        
        # Perform OCR and display/save images
        extracted_text = extract_text_from_image_pdf(pdf_path, display_images, save_images)
        documents = [{"text": extracted_text}]
        st.write("Text extracted from images using Tesseract OCR:")
        st.write(extracted_text)
    else:
        st.write("Text extracted from the PDF:")
        st.write(documents[0]["text"])
    
    # Text splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    text_chunks = splitter.split_documents(documents)

    # Vectorstore creation with embedding and retriever setup
    persist_directory = "doc_db"
    vectorstore = Chroma.from_documents(
        documents=text_chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )
    retriever = vectorstore.as_retriever()

    # QA Chain setup
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # User query input
    query = st.text_input("Ask a question about the HDFC24 document")

    if query:
        # Get response from the QA system
        response = qa_chain.invoke({"query": query})

        # Display the result
        st.subheader("Response")
        st.write(response["result"])

        # Show source documents
        if st.checkbox("Show Source Documents"):
            source_docs = response["source_documents"]
            st.write(source_docs)
else:
    st.error("HDFC24 PDF file not found. Please ensure the file is available.")
