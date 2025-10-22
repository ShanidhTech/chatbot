import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    JSONLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import settings
from db import get_db
from models import Chat
from datetime import datetime

# ================== Embeddings ==================

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# ================== FastAPI App ==================
app = FastAPI(title="Care Home Document QA API")

# ================== Vectorstore ==================
def init_vectorstore():
    """Initialize or reinitialize the Chroma vectorstore."""
    os.makedirs(settings.VECTORSTORE_DIR, exist_ok=True)
    return Chroma(
        persist_directory=settings.VECTORSTORE_DIR,
        embedding_function=embeddings,
        collection_name=settings.COLLECTION_NAME,
    )

# Initial vectorstore and QA chain
vectorstore = init_vectorstore()

# Custom prompt template
prompt_template = """
You are a helpful assistant answering strictly based on the provided context.
If the context does not contain the answer, respond with: "I’m sorry, I don’t have that information in the documents."

Context:
{context}

Question:
{question}

Answer:
"""

# Basic QA chain without custom prompt
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
#     return_source_documents=True
# )

# Using custom prompt template in the QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
    chain_type_kwargs={"prompt": PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )},
    return_source_documents=True
)

# ================== Routes ==================
@app.post("/upload_docs/")
async def upload_docs(files: List[UploadFile] = File(...)):
    """
    Upload and index care home documents such as policies, resident records, etc.
    Supports: PDF, DOCX, TXT, CSV, JSON
    """
    documents = []

    for file in files:
        file_ext = file.filename.lower().split(".")[-1]
        file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            if file_ext == "pdf":
                loader = PyPDFLoader(file_path)
            elif file_ext == "docx":
                loader = Docx2txtLoader(file_path)
            elif file_ext == "txt":
                loader = TextLoader(file_path)
            elif file_ext == "csv":
                loader = CSVLoader(file_path)
            elif file_ext == "json":
                loader = JSONLoader(file_path)
            else:
                os.remove(file_path)
                continue  # skip unsupported files

            docs = loader.load()
            documents.extend(docs)

        except Exception as e:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=f"Error loading {file.filename}: {str(e)}")
        finally:
            os.remove(file_path)

    if not documents:
        raise HTTPException(status_code=400, detail="No valid documents were uploaded.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    vectorstore.add_documents(chunks)

    # Persist only if Chroma supports it
    try:
        vectorstore.persist()
    except Exception:
        pass

    return {"message": f"Uploaded and indexed {len(chunks)} chunks."}


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
    Completely clear ChromaDB collection safely and reinitialize QA chain.
    """
    global vectorstore, qa

    try:
        # Access the underlying Chroma collection
        collection = vectorstore._collection

        # Delete all data in the collection (reset it)
        collection.delete(where={"id": {"$ne": ""}})  # remove all docs (if version supports filtering)
    except Exception:
        try:
            # If delete(where=...) is restricted, just drop the collection and recreate
            client = vectorstore._client
            client.delete_collection(name=settings.COLLECTION_NAME)
            vectorstore = client.create_collection(name=settings.COLLECTION_NAME)
        except Exception:
            pass

    # --- Reinitialize the LangChain Chroma wrapper ---
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

