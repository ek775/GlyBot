"""Helper Script for initializing the Vector Store"""

# imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.ingestion.cache import IngestionCache
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore

import qdrant_client
import os

class QdrantSetup:
    """
    Helper class for initializing the Vector Store
    """
    def __init__(self, data_dir: str = './textbook_text_data/', 
                 cache: str = 'llama-3-cache', 
                 name: str = 'llama-3-cache'):
        """
        Setting Params
        """
        self.data_dir = data_dir
        self.cache = cache
        self.name = name
        self.sync_client = None
        self.async_client = None
        self.vector_store = None
        self.documents = None
        self.index = None
        if os.path.exists(f"./{cache}/qdrant.db"):
            print("Connecting to existing vector DB...")
            self._set_client()
            self.vector_store = self._build_vector_store(use_async=True)
            self.documents = self.read_data(loc=self.data_dir)
            self.index = self._load_index(vector_store=self.vector_store)
        else:
            print("Setting up new vector DB...")
            self._set_client()
            self.vector_store = self._build_vector_store(use_async=False)
            self.documents = self.read_data(loc=self.data_dir)
            self._initialize_vector_db(
                documents=self.documents, 
                cache=self.cache, 
                name=self.name
                )
            print("Connecting async client to DB...")
            self.vector_store = self._build_vector_store(use_async=True)
            self.index = self._load_index(vector_store=self.vector_store)

    def _set_client(self):
        """
        Use sync client to load data, async for db search
        """
        self.sync_client = qdrant_client.QdrantClient(
            path=f"./{self.cache}/qdrant.db",  
            port=6333, 
            grpc_port=6334, 
            prefer_grpc=True)
        self.async_client = qdrant_client.AsyncQdrantClient(
            path=f"./{self.cache}/qdrant.db", 
            port=6333, 
            grpc_port=6334, 
            prefer_grpc=True)
        
    def _build_vector_store(self, use_async: bool):
        """
        build vector store with synchronous and asynchronous clients
        """
        if use_async==False:
            vector_store = QdrantVectorStore(
                collection_name="glyco_store",
                client=self.sync_client,
                prefer_grpc=True
            )
        else:
            vector_store = QdrantVectorStore(
                collection_name="glyco_store",
                aclient=self.async_client,
                prefer_grpc=True
            )
        return vector_store
    
    def read_data(self, loc: str):
        """
        Read data from location
        """
        reader = SimpleDirectoryReader(input_dir=loc, filename_as_id=True)
        documents = reader.load_data()
        return documents
        
    def _load_index(self, vector_store):
        """
        Load index from Vector Store
        """
        # create index
        print("Creating index...")
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        print("Done!")
        return index
    
    def _initialize_vector_db(self, documents, cache: str, name: str):
        """
        Initializing the Vector Store
        """
        # pipeline
        def build_pipeline(vector_store: QdrantVectorStore):
            """builds pipeline, checks for cached data"""
            pipeline = IngestionPipeline(
                    name=name,
                    transformations=[
                        SentenceSplitter(),
                        TitleExtractor(),
                        Settings.embed_model
                    ],
                    docstore=SimpleDocumentStore(),
                    vector_store=vector_store,
                    cache=IngestionCache(collection=f"./{cache}/{name}")
                )
            return pipeline

        # initialize client and vector store
        pipeline = build_pipeline(vector_store=self.vector_store)
        pipeline.run(documents=documents) 