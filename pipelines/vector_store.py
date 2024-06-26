"""Helper Script for initializing the Vector Store"""

# imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.vector_stores.qdrant import QdrantVectorStore

import qdrant_client

def initialize_vector_db(data_dir: str = './textbook_text_data/', cache: str = 'llama-3-cache', name: str = 'llama-3-cache'):
    """
    Helper function for initializing the Vector Store
    """
    # read data
    print("Reading data...")
    reader = SimpleDirectoryReader(input_dir=data_dir)
    documents = reader.load_data()

    # initialize client and vector store
    print("Initializing Vector DB...")
    client = qdrant_client.QdrantClient(host="localhost", port=6333)
    async_client = qdrant_client.AsyncQdrantClient(location=":memory:")
    vector_store = QdrantVectorStore(
        collection_name="glyco_store",
        client=client,
        aclient=async_client,
        prefer_grpc=True
        )
    
    # pipeline
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(),
            TitleExtractor(),
            Settings.embed_model
        ],
        vector_store=vector_store
    )

    # check for cached vector store
    try:
        pipeline.load(f"./{cache}", cache_name=name)
    except FileNotFoundError:
        # load the vector db
        pipeline.run(documents=documents)
        pipeline.persist(f"./{cache}", cache_name=name)
    # create index
    print("Creating index...")
    index = VectorStoreIndex.from_vector_store(vector_store, use_async=True)
    print("Done!")

    return client, vector_store, index, documents