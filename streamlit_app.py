### Main Application Script for Text Generation Backend ###
import streamlit as st

# import libraries
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
from llama_index.agent.openai import OpenAIAssistantAgent
from qdrant_client import QdrantClient
import numpy as np
# helper scripts
from pipelines.vector_store import QdrantSetup
# other utilities
import re
import os
import sys
import glob
from io import StringIO
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
    with open('./SENSITIVE/google_custom_search.txt', 'r') as f:
        os.environ['GOOGLE_CUSTOM_SEARCH'] = f.read().strip()
# streamlit should serve the keys as environment variables already
except:
    pass

instructions = "You are a glycobiology assistant for GlyGen that helps scientists navigate and utilize a bioinformatics knowledgebase."
prompt_template = (
    "Context information from 'Essentials of Glycobiology' (4th edition) is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Based on the context information provided, assist the user with navigating glygen, accessing data, or answering glycobiology questions, and respond to the query below.\n"
    "Query: {query_str}\n"
    "Answer: "
)
live_prompt_template = PromptTemplate(
    prompt_template, prompt_type=PromptType.CUSTOM,
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

def glyco_essentials_retriever(_index_params: Parameters, _retriever_params: Parameters) -> VectorIndexRetriever:
    db = QdrantSetup(**dict(_index_params.params))
    retriever = VectorIndexRetriever(index=db.index, **dict(_retriever_params.params))
    return retriever

def query_engine_config(_retriever: VectorIndexRetriever, _response_params: Parameters) -> RetrieverQueryEngine:
    response_synthesizer = get_response_synthesizer(**dict(_response_params.params))
    return RetrieverQueryEngine(retriever=_retriever, response_synthesizer=response_synthesizer)

#################################################################################
### Background Knowledge Pipeline from Essentials of Glycobiology 4th ed. ###
#################################################################################

# configure index and retriever for "Essentials of Glycobiology" textbook
# host for local client is the docker container name: qdrant_vector_db OR localhost if not running via compose
index_params = Parameters({
        "use_async": False,
        "local_client": QdrantClient("http://qdrant_vector_db:6333"),
        "data_dir": './textbook_text_data/',
        "cache": cache,
        "name": name
        })

retriever_params = Parameters({
        "similarity_top_k": 5
        })

glyco_retriever = glyco_essentials_retriever(_index_params=index_params, _retriever_params=retriever_params)

# config summary engine for "Essentials of Glycobiology" textbook
# """response_params = Parameters({
#        "response_mode": "tree_summarize",
#         "summary_template": live_prompt_template
#         })
#
# glyco_engine = query_engine_config(_retriever=glyco_retriever, _response_params=response_params)"""
    
#################################################################################
### GlyBot Config, Tools, and Caching ###
#################################################################################

# import tools
from assistant_tools.pubmed_abstracts import build_pubmed_search_tool
from assistant_tools.glygen_pages_crawl import build_google_search_tool
from assistant_tools.glygen_image_api import build_glygen_image_search_tool

@st.cache_resource
def load_tools() -> list:
    """
    Cache the tools separately so they are not rebuilt when building an agent after failed loading.
    TODO: Add more tools as they are built.
    """
    print("Loading Tools")
    tools = [
        build_google_search_tool(),
        build_glygen_image_search_tool(),
        build_pubmed_search_tool(),
    ]
    return tools

# build agent
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

    TODO: Single agent can only be assigned one thread at a time, users loading the app simulataneously currently
    leads to conflicts. Ideally handle without creating endless agents.
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
    TODO: Implement a way to record feedback from the user.
    """
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

def detect_relevance(query:str, threshold:float) -> bool:
    """
    Function for detecting if a user prompt is relevant to the assistant's scope.
    """
    relevance = None

    # use retrieval score to determine relevance semantically
    relevant_textbook = glyco_retriever.retrieve(query)
    nom_relevance = relevant_textbook[0].score

    if nom_relevance < threshold:
        relevance = False
    else:
        relevance = True

    return relevance, relevant_textbook

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
    log_number = ''.join([str(np.random.randint(0,10)) for n in range(12)])
    with open(f"./logging/{log_number}.txt", "x") as f:
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
    st.title(f"{agent_name} \n *A Protoype Glycobiology Assistant*")
    glygen_logo = Image.open("./glygen_logo.png")
    buf = BytesIO()
    glygen_logo.save(buf, format="PNG")
    byt_img = buf.getvalue()
    st.image(byt_img, use_column_width=True)
    st.page_link(page="https://www.glygen.org", label="GlyGen Homepage", icon="🌎")     
    st.divider()   
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
            # check relevance of user query
            relevance, context = detect_relevance(query=prompt, threshold=0.3)
            if relevance == False:
                response = "I'm sorry, I'm not sure how to help with that. Please try asking a question related to glycobiology."

            else:
                augmented_prompt = prompt_template.format(context_str=''.join([c.text for c in context]), query_str=prompt)
                response = agent.chat(augmented_prompt)
            
            # imgfiles = glob.glob("glycan_image_temp_file_*.png")
            # if len(imgfiles) > 0:
            #     print("uploading:",imgfiles)
            #     agent.upload_files(imgfiles)
            # for fn in imgfiles:
            #     os.remove(fn)
        return response
    
    with st.spinner("Thinking..."):
        # hide tracebacks from users
        try:
            response = get_response(prompt)
            stream_capture.flush()
            tool_call = stream_capture.getvalue()
        except Exception as e:
            print(e)
            response = "An error occurred while processing your request. Please try again later."
            stream_capture.flush()
            tool_call = stream_capture.getvalue()

            # save traceback to log file so we can debug
            log_number = ''.join([str(np.random.randint(0,10)) for n in range(12)])
            with open(f"./logging/{log_number}.txt", "x") as f:
                f.write(prompt)
                f.write("\n")
                f.write(tool_call)
                f.write("\n")
                f.write(str(e))
                f.write("\n")
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
