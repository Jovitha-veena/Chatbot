# scripts/embed_pdf.py
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

# Load PDF
loader = PyMuPDFLoader(r"C:\Users\Jerome Vinodh\Documents\Tribal Chatbot\Tribes_in_India_Chatbot_Document.pdf")
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Create local embedding
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store in FAISS vector DB
db = FAISS.from_documents(chunks, embeddings)
db.save_local("vectorstore")

print("✅ Vector store saved.")
