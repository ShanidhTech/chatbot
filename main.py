import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File
from db import get_session
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from config import settings


from fastapi import Depends
from models import Chat
from sqlmodel import Session

# ========== FASTAPI ==========
app = FastAPI(title="Document QA API with Gemini")


# ========== LangChain Setup ==========
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = Chroma(
    persist_directory=settings.VECTORSTORE_DIR,
    embedding_function=embeddings,
    collection_name=settings.COLLECTION_NAME
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True
)


# ========== ROUTES ==========

@app.post("/upload_docs/")
async def upload_docs(files: List[UploadFile] = File(...)):
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


# @app.post("/ask/")
# async def ask_question(question: str):
#     result = qa({"query": question})
#     answer = result["result"]
#     sources = [
#         {"source": doc.metadata.get("source", "unknown"), "snippet": doc.page_content[:200]}
#         for doc in result.get("source_documents", [])
#     ]
#     return {"answer": answer, "sources": sources}


@app.post("/ask/")
async def ask_question(question: str):
    result = qa({"query": question})
    answer = result["result"]
    chat = Chat(question=question, answer=answer)
    with next(get_session()) as session:
        session.add(chat)
        session.commit()
        session.refresh(chat)
    return {"id": chat.id, "answer": answer}