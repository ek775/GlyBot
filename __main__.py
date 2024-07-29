### Main Application Script for Text Generation Backend ###
"""
Open Issues:

1. Core RAG eval pipeline causes multi-client access of local Qdrant

2. OpenAPI tool for GlyGen API needs swagger 2.0 converted to OpenAPI 3.0+
2A. BEFORE TESTING: Consider Safety of API - LLM generated queries may be harmful

3. Google Search tool

"""

# import libraries
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, get_response_synthesizer, VectorStoreIndex
from llama_index.core.response_synthesizers.base import BaseSynthesizer
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

# CLI args
# model: openai or ollama
try:
    assert sys.argv[1] in ['openai', 'ollama'], "Please specify the LLM to employ: 'openai' or 'ollama'"
    llm = sys.argv[1]

except IndexError:
    llm = 'openai'

# mode: eval or chat
try:
    assert sys.argv[2] in ['eval', 'chat'], "Please specify the mode of operation: 'eval' or 'chat'"
    mode = sys.argv[2]
    # log if eval mode
    if mode == 'eval':
        print("logging debug output")
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

except IndexError:
    mode = 'chat'

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

# config tools

class Parameters:
    """Base class for setting parameters"""
    def __init__(self, param_dict: dict):
        """Set parameters from a dictionary"""
        for i in param_dict:
            setattr(self, i, param_dict[i])
    @property
    def params(self):
        for attr in self.__dict__:
            value = getattr(self, attr)
            yield attr, value

class QueryEngineConfig:
    """
    Configures the Query Engine.

    Full implementation will include multi-parameter testing using a grid-search approach.
    """
    @property
    def query_engine(self) -> RetrieverQueryEngine:
        """
        Initialize a single query engine. 

        Note that only one client can access the datase at the same time.Initializing the query engine will 
        lock the database without server infrastructure.
        """
        db = QdrantSetup(use_async=self.index_params.use_async,
                         data_dir=self.index_params.data_dir,
                         cache=self.index_params.cache,
                         name=self.index_params.name
                         )
        retriever = VectorIndexRetriever(index=db.index,
                                         similarity_top_k=self.retriever_params.similarity_top_k
                                         )
        response_synthesizer = get_response_synthesizer(response_mode=self.response_params.response_mode,
                                                        text_qa_template=self.response_params.text_qa_template
                                                        )
        return RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)

    def __init__(self, 
                 index_params: Parameters, 
                 retriever_params: Parameters, 
                 response_params: Parameters,
                 ):
        self.index_params = index_params
        self.retriever_params = retriever_params
        self.response_params = response_params
        # setup query engine
        self.query_engine

    def grid_search(self, index_params, retriever_params, response_params):
        """Advances configuration to the next set of parameters"""
        pass


#################################################################################
# Evaluation mode
if mode == 'eval':
    print("Running RAGAS Evaluation...")

    # setting parameters
    
    full_index_params = Parameters({
        "use_async": True,
        "data_dir": './textbook_text_data/',
        "cache": cache,
        "name": name
        })
    full_retriever_params_k3 = Parameters({
        "similarity_top_k": 3
        })
    full_retriever_params_k5 = Parameters({
        "similarity_top_k": 5
        })
    full_retriever_params_k7 = Parameters({
        "similarity_top_k": 7
        })
    full_response_params = Parameters({
        "response_mode": "tree_summarize",
        "text_qa_template": live_prompt_template
        })
    dummy_response_params = Parameters({
        "response_mode": "no_text",
        "text_qa_template": live_prompt_template
        })
    
    # configure RAG pipelines
    config_k3 = dict(index_params=full_index_params, 
                     retriever_params=full_retriever_params_k3, 
                     response_params=full_response_params)
    config_k5 = dict(index_params=full_index_params,
                     retriever_params=full_retriever_params_k5,
                     response_params=full_response_params)
    config_k7 = dict(index_params=full_index_params,
                     retriever_params=full_retriever_params_k7, 
                     response_params=full_response_params)
    dummy_config = dict(index_params=full_index_params,
                        retriever_params=full_retriever_params_k3, 
                        response_params=dummy_response_params)
    config_list = [config_k3, config_k5, config_k7, dummy_config]

    # run RAG pipelines on curated questions
    for i, config in enumerate(config_list):
        # configure query engine
        config = QueryEngineConfig(**config)
        query_engine = config.query_engine
        params = [config.index_params.params, config.retriever_params.params, config.response_params.params]
        metadata = {}
        for p in params:
            metadata.update(dict(p))
        print("==========================================================================")
        print("Running RAGAS Evaluation with Config: ", config)
        print("==========================================================================")
        # pass to evaluator
        evaluator = GlyBot_Evaluator(
            curated_q_path='./ground_truth_eval_queries/curated_queries.csv',
            query_engine=query_engine
            )
        print("-----| Loading Prompts |-----")
        evaluator.get_prompts()
        print("-----| Evaluating Responses |-----")
        evaluator.response_evaluation(metadata=metadata)
        print("==========================================================================")
        print(f"Finished {i} of {len(config_list)} Evaluations")
        print("==========================================================================")

    print("***** COMPLETE *****")
    sys.exit(0)

#################################################################################
# configure agent
from llama_index.tools.openapi import OpenAPIToolSpec # glygen api: https://api.glygen.org/swagger.json
from llama_index.tools.google import GoogleSearchToolSpec
from llama_index.readers.papers import PubmedReader
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from pydantic import BaseModel, Field
from llama_index.agent.openai import OpenAIAssistantAgent
import requests
import json

# tools
tool_list = []

# glygen api tool
"""Issues: How to Protect? Convert Swagger into OpenAPI"""
#f = requests.get('https://api.glygen.org/swagger.json').text
#api_spec = json.loads(f)

#GlyGenAPITool = OpenAPIToolSpec(spec=api_spec)
#tool_list.append(GlyGenAPITool)


# google search tool
"""Looks interesting, literally no documentation"""
# GoogleSearchTool = GoogleSearchToolSpec()


# pubmed reader tool
class PubMedQuery(BaseModel):
    """pydantic object describing how to search pubmed to the llm"""
    query: str = Field(..., description="Natural Language Query to search for on Pubmed.")

def pubmed_search(query: str):
    """Retrieves abstracts of relevant papers from PubMed"""
    reader = PubmedReader()
    papers = reader.load_data(search_query=query, max_results=10)
    index = VectorStoreIndex.from_documents(papers)
    retriever = index.as_retriever()
    results = retriever.retrieve(query)
    return [r.get_content() for r in results]

PubmedSearchTool = FunctionTool.from_defaults(
    fn= pubmed_search,
    name="pubmed_search_tool",
    description="Use this tool to search for recent studies on PubMed that are related to the user's query.",
    fn_schema=PubMedQuery
    )
tool_list.append(PubmedSearchTool)


# query engine tool for "Essentials of Glycobiology" textbook
toolmeta = ToolMetadata(
    name="essentials_of_glycobiology",
    description="Use this tool to provide entry-level information on glycobiology from the textbook 'Essentials of Glycobiology'.",
)

# configure query engine
index_params = Parameters({
    "use_async": False,
    "data_dir": './textbook_text_data/',
    "cache": cache,
    "name": name
    })

retriever_params = Parameters({
    "similarity_top_k": 5
    })

response_params = Parameters({
    "response_mode": "tree_summarize",
    "text_qa_template": live_prompt_template
    })

config = QueryEngineConfig(index_params=index_params, retriever_params=retriever_params, response_params=response_params)

# create tool
TextbookQueryEngineTool = QueryEngineTool(query_engine=config.query_engine, metadata=toolmeta) # calls sync client??
tool_list.append(TextbookQueryEngineTool)


# agent
agent = OpenAIAssistantAgent.from_new(
    name="GlyBot",
    instructions=instructions,
    model="gpt-3.5-turbo",
    tools=tool_list,
    verbose=True,
    run_retrieve_sleep_time=1.0
    )

#################################################################################
# **Main Loop** 
if mode == "chat":
    print("==========================================================================")
    print("Hello, I am GlyBot, your glycobiology assistant. How can I help you today?")
    while True:
        print("==========================================================================")
        user_input = input("User: ")
        if user_input == "exit":
            print("Goodbye!")
            break
        else:
            print("==========================================================================")
            print("GlyBot:")
            agent.chat(user_input)
            continue
        

# Evaluation
