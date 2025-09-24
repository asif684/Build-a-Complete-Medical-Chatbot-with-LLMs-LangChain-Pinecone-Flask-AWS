from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from euriai.langchain import create_chat_model   # ✅ Use Euri, not OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

app = Flask(__name__)

# -----------------------------
# Define system prompt here
# -----------------------------
system_prompt = """
You are a Medical Assistant for question-answering tasks. 
Use the provided context to answer questions accurately.
If you do not know the answer, politely say you don't know.
"""

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
EURI_API_KEY = os.environ.get('EURI_API_KEY')   # ✅ Euri key

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["EURI_API_KEY"] = EURI_API_KEY

# Load embeddings + Pinecone index
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ✅ Initialize Euri chat model
chatModel = create_chat_model(
    api_key=EURI_API_KEY,
    model="gpt-4.1-nano",
    temperature=0.7
)

# -----------------------------
# Prompt template with context
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Answer the question based on the following context:\n{context}\n\nQuestion: {input}"),
    ]
)

# RAG pipeline
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# -----------------------------
# Flask routes
# -----------------------------
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)
    response = rag_chain.invoke({"input": msg})
    print("Response:", response["answer"])
    return str(response["answer"])

# -----------------------------
# Run app
# -----------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
