
################################DATABASE CREATION SCRIPT GROQ#################################

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

# Load env vars
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH =  r"D:\ML_Course\ANFA\Edubot_opensource\data\books"


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    for doc in documents:
        doc.metadata["topic"] = "vegetables"
        doc.metadata["source"] = doc.metadata.get("source", "unknown")
        doc.metadata["filename"] = doc.metadata["source"].split("/")[-1]
    print(f"Loaded {len(documents)} documents with metadata.")
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=100, length_function=len, add_start_index=True
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} docs into {len(chunks)} chunks.")
    return chunks

def save_to_pinecone(chunks):
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    index_name = "anf-educational-bot"

    # ✅ Use 384 dimension (because all-MiniLM-L6-v2 has 384)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,  # Match with sentence-transformers/all-MiniLM-L6-v2
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = PineconeVectorStore.from_existing_index(index_name, embeddings)

    db.add_documents(chunks)
    print(f"✅ Saved {len(chunks)} chunks to Pinecone index '{index_name}'")


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_pinecone(chunks)

def main():
    generate_data_store()

if __name__ == "__main__":
    main()
################################DATABASE CREATION SCRIPT GPT#################################

