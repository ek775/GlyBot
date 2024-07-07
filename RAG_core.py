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
from pipelines.vector_store import QdrantSetup
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

#######################################################################################
# choose model at exe, apply settings
cache = None
if llm == 'openai':
    # load sensitive stuffs
    key = None
    with open('./SENSITIVE/ek_llama_index_key.txt', 'r') as f:
        key = f.read().strip()
    # connect to OpenAI
    os.environ['OPENAI_API_KEY'] = key
    Settings.llm = OpenAI(model="gpt-3.5-turbo")
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small", 
        embed_batch_size=100
        )
    cache = 'openai_vector_data'
    name = 'openai_pipeline_cache'

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
    cache = 'llama-3_vector_data'
    name = 'llama-3_pipeline_cache'

# initialize/connect to vector database, load documents, create index
db = QdrantSetup(data_dir='./textbook_text_data/', cache=cache, name=name)
index = db.index
documents = db.documents

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

#################################################################################
# Evaluation mode
if mode == 'eval':
    print("Running Evaluation...")

    # make dummy index for non-RAG comparison
    dummy_db = QdrantSetup(
        data_dir='./dummy_data_directory/',
        cache='dummy_db',
        name='dummy_db'
        )
    dummy_index = dummy_db.index
    dummy_documents = dummy_db.documents
    dummy_engine = dummy_index.as_query_engine(
        llm=Settings.llm,
        response_mode="tree_summarize"
        )
    
    # experimental RAG augmentation eval
    evaluator = GlyBot_Evaluator(
        curated_q_path='./ground_truth_eval_queries/curated_queries.csv',
        documents=documents,
        query_engine=query_engine
        )
    print("Generating Prompts...")
    evaluator.get_prompts()
    print("Evaluating Responses...")
    evaluator.response_evaluation()
    # dummy eval
    print("Evaluating Dummy Responses...")
    evaluator.set_query_engine(dummy_engine)
    evaluator.response_evaluation()
    print("***COMPLETE***")
    sys.exit(0)

#################################################################################
# configure chat engine
"""finish me later once ready for interaction with the user"""

# **Main Loop** 
"""run the chatbot once we get there"""

# Evaluation
