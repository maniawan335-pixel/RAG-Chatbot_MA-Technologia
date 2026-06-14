import os
from typing import List
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import TextLoader, PyPDFLoader

def load_documents(data_dir: str) -> List[Document]:
    """
    Scans the specified data directory and loads all .txt and .pdf files.
    
    Args:
        data_dir (str): Path to the directory containing files to load.
        
    Returns:
        List[Document]: List of loaded LangChain Document objects.
    """
    documents = []
    
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory '{data_dir}' does not exist.")
        return documents
        
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if not os.path.isfile(file_path):
            continue
            
        if file.endswith(".txt"):
            try:
                print(f"Loading text file: {file_path}")
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading text file {file_path}: {e}")
        elif file.endswith(".pdf"):
            try:
                print(f"Loading PDF file: {file_path}")
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading PDF file {file_path}: {e}")
                
    return documents

def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Splits a list of documents into chunks using RecursiveCharacterTextSplitter.
    
    Args:
        documents (List[Document]): The list of documents to split.
        chunk_size (int): Max size of each chunk.
        chunk_overlap (int): Overlap between adjacent chunks.
        
    Returns:
        List[Document]: List of split document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    # Test document loader
    docs = load_documents("./data")
    chunks = split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")
