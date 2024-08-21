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
from asyncio import Semaphore
import asyncio
from typing import Optional

class QdrantSetup:
    """
    Helper class for initializing the Vector Store
    """
    def __init__(self,
                 use_async: bool,
                 local_client: Optional[qdrant_client.QdrantClient] = None, 
                 data_dir: str = './textbook_text_data/', 
                 cache: str = 'llama-3-cache', 
                 name: str = 'llama-3-cache',
                 ):
        """
        Setting Params
        """
        # external params
        self.use_async = use_async
        self.data_dir = data_dir
        self.cache = cache
        self.name = name
        # internal params
        self.client = None
        self.vector_store = None
        self.documents = None
        self.index = None
        # initialize
        # with local server client from docker image... WIP
        if local_client is not None:
            self.client = local_client
            self.vector_store = QdrantVectorStore(
                collection_name="glyco_store",
                client=self.client,
            )
            self.documents = self.read_data(loc=self.data_dir)
            self.index = self._load_index(vector_store=self.vector_store)
        # with existing vector store
        elif os.path.exists(f"./{cache}/qdrant.db"):
            print("Connecting to existing vector DB...")
            self._set_client(use_async=self.use_async)
            self._connect_to_vector_store(use_async=self.use_async)
            self.documents = self.read_data(loc=self.data_dir)
            self.index = self._load_index(vector_store=self.vector_store)
        # from scratch with new params
        else:
            print("Setting up new vector DB...")
            self._set_client(use_async=self.use_async)
            # use async client to load data
            self._connect_to_vector_store(use_async=True)
            self.documents = self.read_data(loc=self.data_dir)
            self._initialize_vector_db(
                documents=self.documents, 
                cache=self.cache, 
                name=self.name
                )
            # reconnect and build index using desired client
            self.client = None
            self.vector_store = None
            self._set_client(use_async=self.use_async)
            self._connect_to_vector_store(use_async=self.use_async)
            self.index = self._load_index(vector_store=self.vector_store)

    def _set_client(self, use_async: bool = True):
        """
        Use sync client to load data, async for db search
        """
        if use_async == False:
            self.client = qdrant_client.QdrantClient(
                path=f"./{self.cache}/qdrant.db",  
                port=6333, 
                grpc_port=6334, 
                prefer_grpc=True)
        else:
            self.client = qdrant_client.AsyncQdrantClient(
                path=f"./{self.cache}/qdrant.db", 
                port=6343, 
                grpc_port=6344, 
                prefer_grpc=True)
        
    def _connect_to_vector_store(self, use_async: bool):
        """
        build vector store with synchronous and asynchronous clients
        """
        if use_async == True:
            self.vector_store = QdrantVectorStore(
                collection_name="glyco_store",
                aclient=self.client,
                prefer_grpc=True
            )
        else:
            self.vector_store = QdrantVectorStore(
                collection_name="glyco_store",
                client=self.client,
                prefer_grpc=True
            )
    
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
                        SentenceSplitter(
                            chunk_size=240,
                            chunk_overlap=120
                        ),
                        TitleExtractor(
                            llm=Settings.llm,
                            nodes=1
                        ),
                        Settings.embed_model
                    ],
                    docstore=SimpleDocumentStore(),
                    vector_store=vector_store,
                    cache=IngestionCache(collection=f"./{cache}/{name}")
                )
            return pipeline

        # async loading data
        async def async_load(pipeline, docs, semaphore):
            async with semaphore:
                await pipeline.arun(documents=docs, num_workers=1)
        
        # build pipeline and load data
        pipeline = build_pipeline(vector_store=self.vector_store)
        semaphore = Semaphore(value = 1)

        asyncio.run(async_load(pipeline=pipeline, docs=documents, semaphore=semaphore))
        