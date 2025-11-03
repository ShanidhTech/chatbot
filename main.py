import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import Ollama  # Replace with your LLM
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader, JSONLoader
from langchain.prompts import PromptTemplate
from config import settings

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
app = FastAPI(title="RAG Chatbot: all-MiniLM-L6-v2 Embeddings")

# ======================================================
# EMBEDDINGS + VECTORSTORE
# ======================================================
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# vectorstore = Chroma(
#     persist_directory=VECTORSTORE_DIR,
#     collection_name=COLLECTION_NAME,
#     embedding_function=embedding,
# )

def get_vectorstore(org_id: str):
    print(f"Using vectorstore for org_id: {COLLECTION_NAME}_{org_id}")
    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        collection_name=f"{COLLECTION_NAME}_{org_id}",
        embedding_function=embedding,
    )

# ======================================================
# LLM
# ======================================================

# llm = Ollama(model="phi3:mini")
# llm = Ollama(model="mistral:latest") 
llm = Ollama(
    model="phi3:mini",
    # model="mistral:latest",
    base_url=settings.OLLAMA_BASE_URL
)


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
async def upload_file(org_id: str, file: UploadFile = File(...)):
    try:
        vectorstore = get_vectorstore(org_id)
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
async def ask_question(org_id: str, query: str):
    try:
        vectorstore = get_vectorstore(org_id)
        # Retrieve top 3 docs with scores
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=3)

        # Normalize scores to [0,1]
        relevant_docs = [doc for doc, score in docs_and_scores if (score + 1) / 2 >= 0.3]

        if not relevant_docs:
            return {"answer": "I don’t know based on the uploaded documents.", "sources": []}

        # Custom prompt to restrict LLM to context only
        custom_prompt = PromptTemplate.from_template("""
        You are a helpful assistant that answers ONLY based on the provided context below.
        If the answer is not in the context, reply exactly:
        "I don’t know based on the uploaded documents."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type_kwargs={"prompt": custom_prompt},
            return_source_documents=True,
        )

        result = qa_chain({"query": query})

        return {
            "answer": result["result"].strip(),
            "sources": [doc.metadata.get("source", "N/A") for doc in result["source_documents"]],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================
# Clear vectorstore
# ======================================================
@app.delete("/clear")
async def clear_vectorstore(org_id: str):
    try:
        vectorstore = get_vectorstore(org_id)
        collection = vectorstore._collection
        ids = collection.get(ids=None)["ids"]
        if ids:
            collection.delete(ids=ids)
        return {"message": f"All documents deleted for org_id '{org_id}'"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")
