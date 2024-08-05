### Main Application Script for Text Generation Backend ###
import streamlit as st
st.title("GlyBot: Prototype Glycobiology Assistant")

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
from pydantic.v1.main import BaseModel
# helper scripts
from pipelines.vector_store import QdrantSetup
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
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
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

# custom configs

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
    """
    @property
    def query_engine(self) -> RetrieverQueryEngine:
        """
        Initialize a single query engine. 

        Note that only one client can access the datase at the same time. Initializing the query engine will 
        lock the database without server infrastructure.
        """
        db = QdrantSetup(**dict(self.index_params.params))
        retriever = VectorIndexRetriever(index=db.index, **dict(self.retriever_params.params))
        response_synthesizer = get_response_synthesizer(**dict(self.response_params.params))
        return RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)

    def __init__(self, 
                 index_params: Parameters, 
                 retriever_params: Parameters, 
                 response_params: Parameters,
                 ):
        self.index_params = index_params
        self.retriever_params = retriever_params
        self.response_params = response_params
        # lazy init query engine


#################################################################################
# configure agent
from llama_index.readers.papers import PubmedReader
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from pydantic import BaseModel, Field
from llama_index.agent.openai import OpenAIAssistantAgent
import urllib.parse
from llama_index.core import SummaryIndex
from llama_index.readers.web import SimpleWebPageReader

# tools
tool_list = []


### google search tool @glygen

# load google api key
with open('./SENSITIVE/google_api_key.txt', 'r') as f:
    google_custom_search_key = f.read().strip()

# define tool from function
class GlyGenGoogleSearch(BaseModel):
    """pydantic object describing how to get information from GlyGen to the llm"""
    query: str = Field(..., description="Natural Language Query to search GlyGen web pages and API docs for.")

def glygen_google_search(query: str,
                         key: str = google_custom_search_key, 
                         engine: str = 'e41846d71c58e4f2a',
                         num: int = 5):
    """
    Searches the GlyGen website to find relevant information for navigating the site
    and using the GlyGen APIs to access data.
    """
    url_template = ("https://www.googleapis.com/customsearch/v1?key={key}&cx={engine}&q={query}")
    url = url_template.format(key=key, engine=engine, query=urllib.parse.quote_plus(query))

    if num is not None:
        if not 1 <= num <= 10:
            raise ValueError("num should be an integer between 1 and 10, inclusive")
        url += f"&num={num}"

    # turn search results into documents
    documents = SimpleWebPageReader(html_to_text=True).load_data([url])

    # sift through the response to get the relevant information
    index = SummaryIndex.from_documents(documents)
    retriever = index.as_retriever()
    results = retriever.retrieve(query)
    return [r.get_content() for r in results]

GoogleSearchTool = FunctionTool.from_defaults(
    fn = glygen_google_search,
    name = "glygen_google_search_tool",
    description="Use this tool to search the GlyGen website for information on how to Navigate GlyGen and use the GlyGen APIs.",
    fn_schema = GlyGenGoogleSearch)

tool_list.append(GoogleSearchTool)


### pubmed reader tool
class PubMedQuery(BaseModel):
    """pydantic object describing how to search pubmed to the llm"""
    query: str = Field(..., description="Natural Language Query to search for on Pubmed.")

def pubmed_search(query: str):
    """Retrieves abstracts of relevant papers from PubMed"""
    reader = PubmedReader()
    papers = reader.load_data(search_query=query, max_results=5)
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


### query engine tool for "Essentials of Glycobiology" textbook
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


### agent ###
print("Creating Agent")
agent = OpenAIAssistantAgent.from_new(
    name="GlyBot",
    instructions=instructions,
    model="gpt-3.5-turbo",
    tools=tool_list,
    verbose=True,
    run_retrieve_sleep_time=1.0
    )

#################################################################################
# **Main Loop** --> port chatting to streamlit app
if mode == "chat":

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    response = "Hello, I am GlyBot. I am here to help you with your glycobiology questions."

    if prompt := st.chat_input("How can I help you?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Get assistant response
        response = agent.chat(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})