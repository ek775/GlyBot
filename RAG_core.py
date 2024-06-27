### Main Application Script for Text Generation Backend ###

# import libraries
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
# helper scripts
from pipelines.vector_store import initialize_vector_db
from evaluation.response import GlyBot_Evaluator
# other utilities
import os
import sys
import nest_asyncio
nest_asyncio.apply()

# CLI args
assert sys.argv[1] in ['openai', 'ollama'], "Please specify the LLM to employ: 'openai' or 'ollama'"
assert sys.argv[2] in ['eval', 'chat'], "Please specify the mode of operation: 'eval' or 'chat'"
llm = sys.argv[1]
mode = sys.argv[2]

# choose model at exe, apply settings
cache = None
if llm == 'openai':
    # load sensitive stuffs
    key = None
    with open('./SENSITIVE/ek_llama_index_key.txt', 'r') as f:
        key = f.read().strip()
    # connect to OpenAI
    os.environ['OPENAI_API_KEY'] = key
    Settings.llm = OpenAI(model="gpt-4")
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small", 
        embed_batch_size=100
        )
    cache = 'vector_store_cache'
    name = 'llama_cache'

elif llm == 'ollama':
    # NOTE: This option requires running the LLM server locally
    # Install from https://github.com/ollama/ollama.git
    #os.system('ollama serve llama3')
    local_url = "http://localhost:11434" # defaults to this location
    # 8B model, 4.7GB base, docs suggest ~8GB RAM, GPU ideally
    Settings.llm = Ollama(model="llama3",
                          base_url=local_url,
                          request_timeout=180
                          )
    Settings.embed_model = OllamaEmbedding(
        model_name="llama3",
        base_url=local_url,
        ollama_additional_kwargs={"mirostat": 0},
        )
    cache = 'llama-3-cache'
    name = 'llama-3-cache'

# initialize db
index, documents, client = initialize_vector_db(
    data_dir='./textbook_text_data/',
    cache=cache,
    name=name)

# configure retriever and query engine
print("Configuring Query Engine...")
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
if mode == 'eval':
    print("Running Evaluation...")
    evaluator = GlyBot_Evaluator(
        curated_q_path='./ground_truth_eval_queries/curated_queries.csv',
        documents=documents,
        query_engine=query_engine
    )
    print("Generating Prompts...")
    evaluator.get_prompts()
    print("Evaluating Responses...")
    evaluator.response_evaluation()
    sys.exit(0)

# configure chat engine
"""finish me later once ready for interaction with the user"""

# **Main Loop** 
"""run the chatbot once we get there"""

# Evaluation
