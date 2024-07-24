### Main Application Script for Text Generation Backend ###

# import libraries
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, get_response_synthesizer, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
# helper scripts
from pipelines.vector_store import QdrantSetup
from evaluation.response import GlyBot_Evaluator
# other utilities
import os
import sys
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# CLI args
assert sys.argv[1] in ['openai', 'ollama'], "Please specify the LLM to employ: 'openai' or 'ollama'"
assert sys.argv[2] in ['eval', 'chat'], "Please specify the mode of operation: 'eval' or 'chat'"
llm = sys.argv[1]
mode = sys.argv[2]

#######################################################################################
# choose model at exe, apply settings
instructions = "You are a glycobiology assistant for GlyGen that helps scientists navigate and utilize a bioinformatics knowledgebase."
prompt_template = (
    "Context information from 'Essentials of Glycobiology' (4th edition) is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the information from the selected text and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)
live_prompt_template = PromptTemplate(
    prompt_template, prompt_type=PromptType.CUSTOM
)

if llm == 'openai':
    # load sensitive stuffs
    key = None
    with open('./SENSITIVE/ek_llama_index_key.txt', 'r') as f:
        key = f.read().strip()
    # connect to OpenAI
    os.environ['OPENAI_API_KEY'] = key
    Settings.llm = OpenAI(model="gpt-3.5-turbo", system_prompt=instructions)
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
                          request_timeout=180,
                          system_prompt=instructions
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

response_synthesizer = get_response_synthesizer(response_mode="tree_summarize", text_qa_template=live_prompt_template)

query_engine = RetrieverQueryEngine(
    retriever=retriever, 
    response_synthesizer=response_synthesizer
    )

#################################################################################
# Evaluation mode
if mode == 'eval':
    print("Running RAGAS Evaluation...")

    # make dummy index for baseline comparison
    print("Setting up Dummy Index...")
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
# configure agent
from llama_index.tools.openapi import OpenAPIToolSpec # glygen api: https://api.glygen.org/swagger.json
from llama_index.tools.google import GoogleSearchToolSpec
from llama_index.readers.papers import PubmedReader
from llama_index.core.tools import FunctionTool, QueryEngineTool
from pydantic import BaseModel, Field
from llama_index.agent.openai import OpenAIAgent
import requests
#import yaml

# tools
tool_list = []

# glygen api tool
#f = requests.get('https://api.glygen.org/swagger.json').text
#api_spec = yaml.safe_load(f)
GlyGenAPITool = OpenAPIToolSpec(url='https://api.glygen.org/swagger.json')
tool_list.append(GlyGenAPITool)

# google search tool
# GoogleSearchTool = GoogleSearchToolSpec("""NEED WRAPPER, API KEY???""")

# pubmed reader tool
class PubMedQuery(BaseModel):
    """pydantic object describing how to search pubmed to the llm"""
    query: str = Field(..., description="Natural Language Query to search for on Pubmed.")

def pubmed_search(query: str):
    """Retrieves abstracts of relevant papers from PubMed"""
    reader = PubmedReader()
    papers = reader.load_data(search_query=query, max_results=20)
    index = VectorStoreIndex.from_documents(papers)
    retriever = index.as_retriever()
    results = retriever.retrieve(query)
    return [r.get_content() for r in results]

PubmedSearchTool = FunctionTool.from_defaults(
    fn= pubmed_search,
    name="PubmedSearchTool",
    description="Use this tool to search for recent studies on PubMed that are related to the user's query.",
    fn_schema=PubMedQuery
    )
tool_list.append(PubmedSearchTool)

# query engine tool for "Essentials of Glycobiology" textbook
TextbookQueryEngineTool = QueryEngineTool(query_engine=query_engine)
tool_list.append(TextbookQueryEngineTool)

# agent
agent = OpenAIAgent(
    llm=Settings.llm,
    tools=tool_list,
    verbose=True
    )

#################################################################################
# **Main Loop** 
if mode == "chat":
    print("Hello, I am GlyBot, your glycobiology assistant. How can I help you today?")
    while True:
        user_input = input("User: ")
        if user_input == "exit":
            print("Goodbye!")
            break
        print("GlyBot: " + str(agent.chat(user_input)))
        

# Evaluation
