### Main Application Script for Text Generation Backend ###
import streamlit as st
st.title("GlyBot: Prototype Glycobiology Assistant")
st.spinner("Loading...")

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
import numpy as np
# helper scripts
from pipelines.vector_store import QdrantSetup
# other utilities
import re
import requests
import os
import sys
from io import StringIO
import base64
#import logging
from PIL import Image
from io import BytesIO
from typing import Optional

#######################################################################################
# apply settings, configure logging
#######################################################################################

#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

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

def query_engine_config(index_params: Parameters, retriever_params: Parameters, response_params: Parameters) -> RetrieverQueryEngine:
    db = QdrantSetup(**dict(index_params.params))
    retriever = VectorIndexRetriever(index=db.index, **dict(retriever_params.params))
    response_synthesizer = get_response_synthesizer(**dict(response_params.params))
    return RetrieverQueryEngine(retriever=retriever, response_synthesizer=response_synthesizer)

#################################################################################
### Tools for the Assistant ###
#################################################################################

### google search tool @glygen
def build_google_search_tool() -> FunctionTool:
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

### glygen image search tool
def build_glygen_image_search_tool() -> FunctionTool:
    class GlyGenImageSearch(BaseModel):
        """pydantic object describing how to get glycan images from GlyGen to the llm"""
        query: str = Field(..., 
                           pattern=r"[A-Z]{1}\d{5}[A-Z]{2}",
                           description="Glytoucan accession ID to query for glycan images.")

    def glygen_image_search(query: str):
        """
        Searches GlyGen for images of glycans based on their GlyTouCan accession ID.
        """
        # check if the query is a GlyToucan ID
        if len(query) != 8 or not query[0].isalpha() or not query[1:6].isnumeric() or not query[6:].isalpha():
            return "The query should be a valid GlyTouCan ID"

        # query the api
        url = f"https://api.glygen.org/glycan/image/{query}"
        response = requests.post(url=url, verify=False)

        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.save("glycan_image_temp_file.png")
            return base64.b64encode(response.content).decode('utf-8')
        elif response.status_code == 404:
            return "No image found for the given GlyTouCan ID"
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
def build_pubmed_search_tool() -> FunctionTool:
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
def build_textbook_tool() -> QueryEngineTool:
    print("Building Textbook Query Engine Tool")

    # start server
    #print("Starting Qdrant Server")
    #os.system("docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest")

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
    
#################################################################################
### GlyBot Config and Caching ###
#################################################################################

@st.cache_resource
def load_tools() -> list:
    """
    Cache the tools separately so they are not rebuilt when building an agent after failed loading.
    """
    print("Loading Tools")
    tools = [
        build_google_search_tool(),
        build_glygen_image_search_tool(),
        build_pubmed_search_tool(),
        build_textbook_tool()
    ]
    return tools

@st.cache_resource
def build_agent(_tools: list,
                openai_assistant_name: Optional[str]="GlyBot", 
                thread_id: Optional[str] = None, 
                assistant_id: Optional[str] = None) -> OpenAIAssistantAgent:
    """
    Connects to the OpenAI API to build an assistant agent with the given name and instructions.

    If an assistant with the given name already exists, it will be loaded.

    If we are making a new assistant, we give it a new name and allow a new one to be created, 
    caching the id for future use.

    Returns: agent
    """
    print("Connecting to OpenAI API")
    try:
        agent = OpenAIAssistantAgent.from_existing(
            assistant_id=assistant_id,
            thread_id=thread_id,
            tools=_tools,
            verbose=True,
            run_retrieve_sleep_time=1.0
        )
        print("Agent Loaded")
        return agent
    except:
        agent = OpenAIAssistantAgent.from_new(
            name=openai_assistant_name,
            instructions=instructions,
            model="gpt-4o-mini-2024-07-18",
            thread_id=thread_id,
            tools=_tools,
            openai_tools=[{"type": "code_interpreter"}, {"type": "file_search"}],
            verbose=True,
            run_retrieve_sleep_time=1.0
            )
        print("Agent Built")
        return agent

#################################################################################
# Additional functions and utilities for app UI
#################################################################################

# custom stream handler to capture tool output from stdout
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

# helper function for recording feedback
def record_feedback(feedback: Optional[str] = None) -> None:
    """ 
    Writes feedback to text local text file line by line. 
    Accessible from "codespace" in deployed streamlit community cloud context.
    """
    if feedback != None:
        with open("feedback.txt", "a") as f:
            f.write(feedback)
            f.write("\n")
        st.write("Thank you for your feedback!")
    else:
        pass

# helper function for extracting urls from tool output
def redirect_entrez_to_pubmed(pmcid:str) -> str:
    """
    Function for redirecting entrez links to pubmed links for the "learn more" button in the chat interface.
    """
    pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmcid}"
    return pubmed_url

def extract_urls(text:str) -> list:
    """
    Function for extracting urls from tool output. 

    Used for the "learn more" button in the chat interface.
    """
    urls = re.findall(r'(https?://\S+)', text)
    unique_urls = list(set(urls))
    valid_urls = []
    for url in unique_urls:
        if str("\\\\") in url:
            continue
        if str("googleapis") in url:
            continue
        if url.endswith(','):
            url = url[:-1]
        if url.endswith('"'):
            url = url[:-1]
        if url.endswith(('/', '.html', '.com', '.org', '.gov')):
            valid_urls.append(url)
        if url.endswith('pmc'):
            pmcid = re.search(r'id=(\d+)', url).group(1)
            valid_urls.append(redirect_entrez_to_pubmed(pmcid=pmcid))

    return valid_urls

#################################################################################
# Streamlit Configuration and Main Application Script
#################################################################################

# Initialize chat history, tool output history, images, urls
if "messages" not in st.session_state:
    st.session_state.messages = []
if "tool_output" not in st.session_state:
    st.session_state.tool_output = []
if "learn_more_urls" not in st.session_state:
    st.session_state.learn_more_urls = ['https://www.glygen.org']
if "glycan_images" not in st.session_state:
    st.session_state.glycan_images = []
 
# initialize agent, cached so not rebuilt on each rerun
# in case of errors, do not show user tracebacks
try:
    # be sure to update id once testing a new agent
    agent_name = "GlyBot-0.2"
    id = 'asst_Kl07X7RbVCbCYtOdqiLqFzQ8'
    tools = load_tools()
    agent = build_agent(
        _tools=tools, 
        openai_assistant_name=agent_name, 
        assistant_id=id
        )
except Exception as e:
    print(e)
    st.write("An error occurred while connecting to the assistant. Please try again later.")
    # save traceback to log file so we can debug
    log_number = str([np.random.randint(0,10) for n in range(12)])
    with open(f"./logging/{log_number}.txt", "w") as f:
        for line in e:
            f.write(line)

# check for redundant starter messages in history
starter_msg = "Hello! I'm GlyBot, your glycobiology assistant. How can I help you today?"
for msg in st.session_state.messages:
    if msg["content"] == starter_msg:
        st.session_state.messages.remove(msg)

# Display chat messages, etc. from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Display additional info and utilities in side bar
with st.sidebar:
    st.page_link(page="https://www.glygen.org", label="GlyGen Homepage", icon="🌎")        
    # feedback form
    with st.popover("Provide Feedback"):
        with st.form(key="feedback", clear_on_submit=True):
            feedback = st.text_area(
                label="Feedback", 
                value=None, 
                placeholder="Please provide feedback on your experience with GlyBot."
                )
            st.form_submit_button("Submit", on_click=record_feedback(feedback))
    
    

# React to user input
response = starter_msg

if prompt := st.chat_input("How can I help you?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # get response from assistant
    # capture tool calls from stdout
    stream_capture = StreamCapture()

    def get_response(prompt):
        with stream_capture:
            response = agent.chat(prompt)
            if os.path.exists("glycan_image_temp_file.png"):
                agent.upload_files(["glycan_image_temp_file.png"])
                os.remove("glycan_image_temp_file.png")
        return response
    
    with st.spinner("Thinking..."):
        # hide tracebacks from users
        try:
            response = get_response(prompt)
            stream_capture.flush()
            tool_call = stream_capture.getvalue()
        except Exception as e:
            print(e)
            response = "An error occurred while processing your request."
            stream_capture.flush()
            tool_call = stream_capture.getvalue()

            # save traceback to log file so we can debug
            log_number = ''.join([str(np.random.randint(0,10)) for n in range(12)])
            with open(f"./logging/{log_number}.txt", "w") as f:
                f.write(prompt)
                f.write("\n")
                f.write(tool_call)
                f.write("\n")
                f.write(e)
        # save tool calls to history
        st.session_state.tool_output.append(tool_call)
        st.session_state.learn_more_urls.extend(extract_urls(tool_call))

        # display tool call popup and links extracted from the tools in the sidebar
        with st.sidebar:
            # links to glygen, etc. from tool output
            st.sidebar.title("Learn More")
            st.session_state.learn_more_urls = list(set(st.session_state.learn_more_urls))
            for url in st.session_state.learn_more_urls:
                st.page_link(page=url, label=url, icon="🔗")
            st.divider()

            # display actual tool output in popup menu in sidebar
            with st.popover("Tool Calls", help="View how GlyBot uses its information tools to give its answer."):
                for i in st.session_state.tool_output:
                    st.text(i)

# Display assistant response in chat message container
with st.chat_message("assistant"):
    st.markdown(response)
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})