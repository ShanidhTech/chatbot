import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, JSONLoader

# ======================================================
# CONFIGURATION
# ======================================================
VECTORSTORE_DIR = "./chroma_db"
UPLOAD_DIR = "./data"
COLLECTION_NAME = "my_uploaded_docs"

os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ======================================================
# FASTAPI APP
# ======================================================
app = FastAPI(title="Local RAG Chatbot: Ollama Embeddings + Mistral")

# ======================================================
# EMBEDDINGS + VECTORSTORE
# ======================================================
# Use Mistral embeddings (4096-dim)
embedding = OllamaEmbeddings(model="mistral")  

vectorstore = Chroma(
    persist_directory=VECTORSTORE_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding,
)

# ======================================================
# LLM
# ======================================================
llm = Ollama(model="mistral")  # Local Mistral 7B

# ======================================================
# Helper function to load documents
# ======================================================
def load_document(file_path: str):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".csv":
        loader = CSVLoader(file_path)
    elif ext == ".json":
        loader = JSONLoader(file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    return loader.load()

# ======================================================
# Upload and index files
# ======================================================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        docs = load_document(file_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        vectorstore.add_documents(chunks)
        vectorstore.persist()
        return {"message": f"File '{file.filename}' uploaded and indexed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======================================================
# Ask questions
# ======================================================
@app.get("/ask")
async def ask_question(query: str):
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
        )
        result = qa_chain({"query": query})
        return {
            "answer": result["result"],
            "sources": [doc.metadata.get("source", "N/A") for doc in result["source_documents"]],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======================================================
# Clear vectorstore
# ======================================================
@app.delete("/clear")
async def clear_vectorstore():
    global vectorstore
    try:
        # Get all document IDs
        all_ids = vectorstore._collection.get(ids=None)["ids"]  # returns list of IDs

        if all_ids:
            vectorstore._collection.delete(ids=all_ids)

        return {"message": "All documents cleared, vectorstore intact."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")
