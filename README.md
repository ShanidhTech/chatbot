# Document QA API with Gemini

A FastAPI-based document question-answering API using **LangChain**, **Google Gemini embeddings**, and **Chroma vector store**. Users can upload PDF or DOCX documents and ask questions to retrieve answers from the uploaded documents. All Q&A pairs are stored in MongoDB.

---

## Features

- Upload PDF and DOCX documents and index them for semantic search.
- Ask questions based on the indexed documents.
- Store Q&A pairs in MongoDB for history tracking.
- Clear and reinitialize the Chroma vector store if needed.

---

## Tech Stack

- **FastAPI** - API framework
- **LangChain** - LLM and retrieval chain
- **Chroma** - Vector database for embeddings
- **Google Gemini** - Embeddings and chat LLM
- **MongoDB** - Storage for Q&A history
- **Python 3.10+**

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

2. **Create and activate a virtual environment:**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Set environment variables in .env:**

```bash
GOOGLE_API_KEY=<your-google-api-key>
GEMINI_API_KEY=<your-google-gemini-key>

VECTORSTORE_DIR=./chroma_db
COLLECTION_NAME=my_uploaded_docs
UPLOAD_DIR=./data

MONGO_URI=mongodb://mongodb:password@localhost:27017/admin
MONGO_DB_NAME=chatbot
```

5. **Run FastAPI server:**

```bash
uvicorn main:app --reload --env-file .env
```