"""Helper Script for initializing the Vector Store"""

# imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.ingestion.cache import IngestionCache
from llama_index.vector_stores.qdrant import QdrantVectorStore

import qdrant_client
import asyncio
import os

class QdrantSetup:
    """
    Helper class for initializing the Vector Store
    """
    def __init__(self, data_dir: str = './textbook_text_data/', 
                 cache: str = 'llama-3-cache', 
                 name: str = 'llama-3-cache',
                 use_async: bool = True):
        """
        Setting Params
        """
        self.data_dir = data_dir
        self.cache = cache
        self.name = name
        self.use_async = use_async
        self.client = self._set_client(use_async)
        self.vector_store = self._build_vector_store(client=self.client)
        self.documents = self.read_data(loc=self.data_dir)
        self.index = None
        if not os.path.exists(f"./{cache}/qdrant.db"):
            self._initialize_vector_db(
                documents=self.documents, 
                cache=self.cache, 
                name=self.name
                )
            self.index = self._load_index(vector_store=self.vector_store)
        else:
            self.index = self._load_index(vector_store=self.vector_store)

    def _set_client(self, use_async):
        """
        Use sync or async client
        """
        if use_async==False:
            client = qdrant_client.QdrantClient(
                path=f"./{self.cache}/qdrant.db",  
                port=6333, 
                grpc_port=6334, 
                prefer_grpc=True)
            return client
        else:
            client = qdrant_client.AsyncQdrantClient(
                path=f"./{self.cache}/qdrant.db", 
                port=6333, 
                grpc_port=6334, 
                prefer_grpc=True)
            return client
        
    def _build_vector_store(self, client):
        """
        build vector store from client
        """
        if client.__class__.__name__ == 'QdrantClient':
            vector_store = QdrantVectorStore(
                collection_name="glyco_store",
                client=client,
                prefer_grpc=True
                )
            return vector_store
        
        elif client.__class__.__name__ == 'AsyncQdrantClient':
            vector_store = QdrantVectorStore(
                collection_name="glyco_store",
                aclient=client,
                prefer_grpc=True
                )
            return vector_store
    
    def read_data(self, loc: str):
        """
        Read data from location
        """
        reader = SimpleDirectoryReader(input_dir=loc)
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
            try:
                pipeline = IngestionPipeline.load(f"./{cache}", cache_name=name)
                return pipeline
            except FileNotFoundError:
                pipeline = IngestionPipeline(
                    transformations=[
                        SentenceSplitter(),
                        TitleExtractor(),
                        Settings.embed_model
                    ],
                    vector_store=vector_store,
                    cache=IngestionCache(collection=f"./{cache}/{name}")
                )
                return pipeline

        # initialize client and vector store
        print("Initializing Vector DB...")
        if self.client.__class__.__name__ == 'QdrantClient':
            pipeline = build_pipeline(vector_store=self.vector_store)
            pipeline.run(documents=documents)
        
        elif self.client.__class__.__name__ == 'AsyncQdrantClient':
            async def run_pipeline(pipeline: IngestionPipeline, documents):
                await pipeline.arun(documents=documents, num_workers=4)
            asyncio.run(
                run_pipeline(
                    pipeline=build_pipeline(vector_store=self.vector_store), documents=documents)
                    )   