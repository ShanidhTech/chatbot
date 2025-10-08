import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, Depends
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from config import settings
from db import get_db
from models import Chat
from datetime import datetime


app = FastAPI(title="Document QA API with Gemini")


# ======== LangChain Setup ========
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
# vectorstore = Chroma(
#     persist_directory=settings.VECTORSTORE_DIR,
#     embedding_function=embeddings,
#     collection_name=settings.COLLECTION_NAME
# )

# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
#     return_source_documents=True
# )


embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Ensure persist directory exists
os.makedirs(settings.VECTORSTORE_DIR, exist_ok=True)

# Create or get collection safely
try:
    vectorstore = Chroma(
        persist_directory=settings.VECTORSTORE_DIR,
        embedding_function=embeddings,
        collection_name=settings.COLLECTION_NAME,
    )
except Exception as e:
    # If collection/table missing, reinitialize empty
    vectorstore = Chroma(
        persist_directory=settings.VECTORSTORE_DIR,
        embedding_function=embeddings,
        collection_name=settings.COLLECTION_NAME,
    )
    vectorstore.persist()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)


# ======== ROUTES ========

@app.post("/upload_docs/")
async def upload_docs(files: List[UploadFile] = File(...)):
    """
    Upload and index documents.
    """
    documents = []
    for file in files:
        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        if file_path.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load_and_split()
        elif file_path.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
        else:
            continue

        documents.extend(docs)
        os.remove(file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_chunks = splitter.split_documents(documents)
    vectorstore.add_documents(docs_chunks)

    try:
        vectorstore.persist()
    except Exception:
        pass

    return {"message": f"Uploaded and indexed {len(docs_chunks)} chunks."}


@app.post("/ask/")
async def ask_question(question: str, db=Depends(get_db)):
    """
    Ask a question and get an answer from the indexed documents.
    Also store the Q&A pair in MongoDB.
    """
    result = qa({"query": question})
    answer = result["result"]

    chat = Chat(question=question, answer=answer, created_at=datetime.utcnow())
    chat_dict = chat.dict(by_alias=True)

    # Insert into MongoDB
    await db["chat"].insert_one(chat_dict)

    return {"id": str(chat_dict["_id"]), "answer": answer}


@app.delete("/clear_chroma/")
async def clear_chroma_db():
    """
    Clear and reinitialize the ChromaDB.
    """
    global vectorstore, qa
    
    db_path = settings.VECTORSTORE_DIR
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        os.makedirs(db_path, exist_ok=True)
    else:
        os.makedirs(db_path, exist_ok=True)
        
    vectorstore = Chroma(
        persist_directory=settings.VECTORSTORE_DIR,
        embedding_function=embeddings,
        collection_name=settings.COLLECTION_NAME,
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
    )
    return {"message": "ChromaDB cleared and reinitialized successfully."}
