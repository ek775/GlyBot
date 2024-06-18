### Main Application Script for Text Generation Backend ###

# import libraries & helper scripts
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import Settings

from evaluation.CRUD import CRUD_metrics

import os
import pandas as pd

# apply settings
os.environ['OPENAI_API_KEY'] = 'YOUR OPENAI API KEY'
Settings.llm = OpenAI(model="gpt-4")
Settings.embed_model = OpenAIEmbedding(
    model="text-embedding-3-small", embed_batch_size=100
)

# initialize objects


# **Main Loop** 