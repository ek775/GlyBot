### Main Application Script for Text Generation Backend ###

# import libraries
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
# helper scripts
from pipelines.vector_store import initialize_vector_db
from evaluation.response import GlyBot_Evaluator
# other utilities
import os
import sys

# load sensitive stuffs
key = None
with open('./SENSITIVE/ek_llama_index_key.txt', 'r') as f:
    key = f.read().strip()

# apply settings
os.environ['OPENAI_API_KEY'] = key
Settings.llm = OpenAI(model="gpt-4")
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small", embed_batch_size=100
)

# initialize db, query engine, chat engine
client, vector_store, index, documents = initialize_vector_db(
    data_dir='./textbook_text_data/')

# configure retriever and query engine
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5
    )
response_synthesizer = get_response_synthesizer(response_mode="tree_summarize")
query_engine = RetrieverQueryEngine(
    retriever=retriever, 
    response_synthesizer=response_synthesizer
    )


# Evaluation mode
if sys.argv[1] == 'eval':
    evaluator = GlyBot_Evaluator(
        curated_q_path='./curated_q.csv',
        documents=documents,
        query_engine=query_engine
    )
    evaluator.transform()
    evaluator.evaluate()
    sys.exit(0)

# configure chat engine
"""finish me later once ready for interaction with the user"""

# **Main Loop** 
"""run the chatbot once we get there"""

# Evaluation
