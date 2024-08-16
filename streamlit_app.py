### Main Application Script for Text Generation Backend ###
import streamlit as st
st.title("GlyBot: Prototype Glycobiology Assistant")

# import libraries
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings, get_response_synthesizer, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.readers.papers import PubmedReader
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from pydantic import BaseModel, Field
from llama_index.agent.openai import OpenAIAssistantAgent
import urllib.parse
from llama_index.core import SummaryIndex
from llama_index.readers.web import SimpleWebPageReader
#from qdrant_client import QdrantClient
# helper scripts
from pipelines.vector_store import QdrantSetup
# other utilities
import requests
import os
import sys
from io import StringIO, BytesIO
from PIL import Image
import base64
import asyncio

#######################################################################################
# apply settings

# load keys
# for testing locally, use the keys in the SENSITIVE folder
try:
    with open('./SENSITIVE/openai_api_key.txt', 'r') as f:
        os.environ['OPENAI_API_KEY'] = f.read().strip()
    with open('./SENSITIVE/google_api_key.txt', 'r') as f:
        os.environ['GOOGLE_API_KEY'] = f.read().strip()
# streamlit should serve the keys as environment variables already
except:
    pass

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
    
Settings.llm = OpenAI(model="gpt-3.5-turbo", system_prompt=instructions)
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small", 
    embed_batch_size=100
    )
cache = 'openai_vector_data'
name = 'openai_pipeline_cache'

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

def query_engine_config(index_params: Parameters, retriever_params: Parameters, response_params: Parameters):
    db = QdrantSetup(**dict(index_params.params))
    retriever = VectorIndexRetriever(index=db.index, **dict(retriever_params.params))
    response_synthesizer = get_response_synthesizer(**dict(response_params.params))
    return RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)

#################################################################################
# configure agent

# build tools

### google search tool @glygen
def build_google_search_tool():
    print("Building Google Search Tool")
    # load google api key
    # handled by streamlit secrets
    
    google_custom_search_key = os.environ['GOOGLE_API_KEY']

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

    print("Google Search Tool Built")
    return GoogleSearchTool


def build_glygen_image_search_tool():
    pass
    class GlyGenImageSearch(BaseModel):
        """pydantic object describing how to get glycan images from GlyGen to the llm"""
        query: str = Field(..., 
                           pattern=r"[A-Z]{1}\d{5}[A-Z]{2}",
                           description="Glytoucan accession ID to query for glycan images.")

    def glygen_image_search(query: str):
        """
        Searches GlyGen for images of glycans based on their GlyToucan accession ID.
        """
        # check if the query is a GlyToucan ID
        if len(query) != 8 or not query[0].isalpha() or not query[1:6].isnumeric() or not query[6:].isalpha():
            return "The query should be a valid GlyToucan ID"

        # query the api
        url = f"https://api.glygen.org/glycan/image/{query}"
        response = requests.post(url=url)

        if response.status_code == 200:
            return base64.b64encode(response.content).decode('utf-8')
        elif response.status_code == 404:
            return "No image found for the given GlyToucan ID"
        else:
            return "An error occurred while fetching the image"
    
    GlyGenImageSearchTool = FunctionTool.from_defaults(
        fn = glygen_image_search,
        name = "glygen_image_search_tool",
        description = "Use this tool to search for glycan images from GlyGen.",
        fn_schema = GlyGenImageSearch)
    
    print("GlyGen Image Search Tool Built")
    return GlyGenImageSearchTool


### pubmed reader tool
def build_pubmed_search_tool():
    print("Building PubMed Search Tool")
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
    print("PubMed Search Tool Built")
    return PubmedSearchTool
    

### query engine tool for "Essentials of Glycobiology" textbook
def build_textbook_tool():
    print("Building Textbook Query Engine Tool")
    toolmeta = ToolMetadata(
        name="essentials_of_glycobiology",
        description="Use this tool to provide entry-level information on glycobiology from the textbook 'Essentials of Glycobiology'.",
    )

    # configure query engine
    index_params = Parameters({
        "use_async": False,
        #"local_client": QdrantClient("http://localhost:6333"),
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

    # create tool
    TextbookQueryEngineTool = QueryEngineTool(
        query_engine=query_engine_config(
            index_params=index_params, 
            retriever_params=retriever_params, 
            response_params=response_params), 
        metadata=toolmeta
        )
    
    print("Textbook Query Engine Tool Built")
    return TextbookQueryEngineTool
    

### agent ###
@st.cache_resource
def build_agent(thread_id: str = None):
    print("Building Agent")
    agent = OpenAIAssistantAgent.from_new(
        name="GlyBot-0.1",
        instructions=instructions,
        model="gpt-4o-mini-2024-07-18",
        thread_id=thread_id,
        tools=[build_google_search_tool(), 
               build_pubmed_search_tool(), 
               build_textbook_tool(), 
               build_glygen_image_search_tool()],
        verbose=True,
        run_retrieve_sleep_time=1.0
        )
    print("Agent Built")
    return agent

# custom stream handler to capture tool output from stdout and stream it back to st.write_stream
class StreamCapture:
    def __init__(self):
        self.stdout = sys.stdout
        self.stream = StringIO()

    def write(self, data):
        self.stream.write(data)
        self.stdout.write(data)

    def flush(self):
        self.stream.flush()
        self.stdout.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout

    def getvalue(self):
        return self.stream.getvalue()

#################################################################################
# **Main Loop** --> port chatting to streamlit app

# Initialize chat history, tool output history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tool_output" not in st.session_state:
    st.session_state.tool_output = []
    
# initialize agent, cached so not rebuilt on each rerun
agent = build_agent()

# Display chat messages, tool calls from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
with st.sidebar:
    st.sidebar.title("Agent Tool Calls")

# React to user input
response = "Hello, I am GlyBot. I am here to help you with your glycobiology questions."


if prompt := st.chat_input("How can I help you?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # get response from assistant
    # caputre tool calls from stdout
    stream_capture = StreamCapture()

    def get_response(prompt):
        with stream_capture:
            response = agent.chat(prompt)
        return response
        
    async def get_response_async(prompt):
        with stream_capture:
            response = await agent.achat(prompt)
        return response
    
    with st.spinner("Thinking..."):
        response = asyncio.run(get_response_async(prompt))
        stream_capture.flush()
        tool_call = stream_capture.getvalue()
        st.session_state.tool_output.append(tool_call)
        with st.sidebar:
            for tool_call in st.session_state.tool_output:
                st.text(tool_call)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})