import os
from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class RetrievalSystem:
    def __init__(self):
        # Paths
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.vector_store_path = os.path.join(self.base_dir, "data", "vector_store")
        
        # Settings matches ingest_data.py
        self.collection_name = "store_assistant"
        self.embedding_model_name = "all-MiniLM-L6-v2"
        
        # Load Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
        
        # Initialize Vector Store
        if os.path.exists(self.vector_store_path):
            self.vector_store = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embeddings,
                collection_name=self.collection_name
            )
        else:
            print(f"Warning: Vector store not found at {self.vector_store_path}")
            self.vector_store = None

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        """
        Search the vector DB for products or training examples relevant to the query.
        """
        if not self.vector_store:
            return []

        try:
            # Similarity Search
            docs = self.vector_store.similarity_search(query, k=top_k)
            
            results = []
            for doc in docs:
                # Convert Document object to simple dict
                item = doc.metadata.copy()
                item['content'] = doc.page_content
                results.append(item)
                
            return results
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
