import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from document_loader import load_documents, split_documents

def get_embeddings(google_api_key: str = None) -> HuggingFaceEmbeddings:
    """
    Returns a local HuggingFace embeddings instance.
    The optional google_api_key argument is kept for backward compatibility but ignored.
    """
    # Using a small, efficient transformer model for embeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_store(data_dir: str, save_path: str, google_api_key: str = None) -> FAISS:
    """
    Loads documents from data_dir, chunks them, computes embeddings, and saves 
    the created FAISS vector store locally.
    
    Args:
        data_dir (str): Directory containing txt/pdf source files.
        save_path (str): Local path where the FAISS index will be saved.
        google_api_key (str): Optional API key.
        
    Returns:
        FAISS: The constructed FAISS vector store.
    """
    print(f"Ingesting documents from '{data_dir}'...")
    documents = load_documents(data_dir)
    if not documents:
        raise ValueError(f"No documents were successfully loaded from '{data_dir}'.")
        
    chunks = split_documents(documents)
    if not chunks:
        raise ValueError("Documents loaded, but no text chunks were generated.")
        
    embeddings = get_embeddings(google_api_key)
    
    print("Generating embeddings and building FAISS index (this may take a moment)...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    print(f"Saving FAISS index locally to '{save_path}'...")
    vector_store.save_local(save_path)
    return vector_store

def load_vector_store(save_path: str, google_api_key: str = None) -> FAISS:
    """
    Loads a locally saved FAISS index.
    
    Args:
        save_path (str): Local path containing the index.faiss file.
        google_api_key (str): Optional API key.
        
    Returns:
        FAISS: Loaded FAISS vector store.
    """
    embeddings = get_embeddings(google_api_key)
    print(f"Loading local FAISS index from '{save_path}'...")
    # allow_dangerous_deserialization=True is required for loading pickles local files
    return FAISS.load_local(
        save_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

def get_or_create_vector_store(data_dir: str, save_path: str, google_api_key: str = None) -> FAISS:
    """
    Ensures a FAISS vector store is available. Loads from local disk if it exists,
    otherwise creates a new one from source documents.
    
    Args:
        data_dir (str): Directory containing source documents.
        save_path (str): Path to store or load the index.
        google_api_key (str): Optional API key.
        
    Returns:
        FAISS: The loaded or newly created FAISS vector store.
    """
    index_file = os.path.join(save_path, "index.faiss")
    if os.path.exists(save_path) and os.path.exists(index_file):
        try:
            return load_vector_store(save_path, google_api_key)
        except Exception as e:
            print(f"Failed to load existing index at '{save_path}': {e}. Recreating...")
            return create_vector_store(data_dir, save_path, google_api_key)
    else:
        print(f"No existing FAISS index found at '{save_path}'. Recreating...")
        return create_vector_store(data_dir, save_path, google_api_key)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    # Simple self-test
    try:
        store = get_or_create_vector_store("./data", "./faiss_index")
        print("FAISS vector store ready!")
    except Exception as err:
        print(f"Vector store verification failed: {err}")
