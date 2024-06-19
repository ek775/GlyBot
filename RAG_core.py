### Main Application Script for Text Generation Backend ###

# import libraries & helper scripts
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

from pipelines.vector_store import initialize_vector_db

#from evaluation.CRUD import CRUD_metrics

import os
import pandas as pd

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
client, vector_store, index = initialize_vector_db(data_dir='./textbook_text_data/')

# **Main Loop** 