"""Script for running Create Read Update Delete Metrics from API/Model Responses"""

# imports
import pandas as pd
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
)

# gpt-4
gpt4 = OpenAI(temperature=0, model="gpt-4")

faithfulness_gpt4 = FaithfulnessEvaluator(llm=gpt4)
relevancy_gpt4 = RelevancyEvaluator(llm=gpt4)
correctness_gpt4 = CorrectnessEvaluator(llm=gpt4)

def create():
    """Continue a passage"""
    pass

def read():
    """Answer questions about glycobiology"""
    pass

def update():
    """Correct a hallucination"""
    pass

def delete():
    """Summarize a chapter"""
    pass

def CRUD_metrics():
    """Run the CRUD metrics"""
    create()
    read()
    update()
    delete()

from llama_index.core.evaluation import BatchEvalRunner

runner = BatchEvalRunner(
    {"faithfulness": faithfulness_gpt4, "relevancy": relevancy_gpt4},
    workers=8,
)

eval_results = await runner.aevaluate_queries(
    vector_index.as_query_engine(llm=llm), queries=qas.questions
)

# If we had ground-truth answers, we could also include the correctness evaluator like below.
# The correctness evaluator depends on additional kwargs, which are passed in as a dictionary.
# Each question is mapped to a set of kwargs
#

# runner = BatchEvalRunner(
#     {"correctness": correctness_gpt4},
#     workers=8,
# )

# eval_results = await runner.aevaluate_queries(
#     vector_index.as_query_engine(),
#     queries=qas.queries,
#     reference=[qr[1] for qr in qas.qr_pairs],
# )