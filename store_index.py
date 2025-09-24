from dotenv import load_dotenv
import os
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore
from euriai.langchain import create_chat_model   # ✅ Import Euri Chat Model

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
EURI_API_KEY = os.environ.get('EURI_API_KEY')   # ✅ Use Euri API key

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["EURI_API_KEY"] = EURI_API_KEY       # ✅ Set Euri API key env

# Load and process documents
extracted_data = load_pdf_file(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# Hugging Face embeddings
embeddings = download_hugging_face_embeddings()

# Pinecone setup
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"  # change if desired

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

# Create vector store
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)

# ✅ Initialize Euri chat model
chat_model = create_chat_model(
    api_key=EURI_API_KEY,
    model="gpt-4.1-nano",   # you can swap with other Euri models
    temperature=0.7
)
