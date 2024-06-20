"""Script for running Create Read Update Delete Metrics from API/Model Responses"""

# imports
import pandas as pd
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    BaseEvaluator
)
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.metrics.critique import harmfulness
from ragas.integrations.llama_index import evaluate

class glybot_response_metrics(BaseEvaluator):
    """
    Class for running response evaluation in the main experimentation pipeline.
    """
    def __init__(self, test_data, query_engine):
        """
        Initialize the evaluator with the test data and query engine.
        """
        self.test_data = test_data
        self.query_engine = query_engine
        self.llm_metrics = {
            FaithfulnessEvaluator(),
            RelevancyEvaluator(),
            CorrectnessEvaluator()
        }
        self.ragas_metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            harmfulness
        ]
        def transform(self):
            """
            Processes the test data for evaluation.
            """
            pass
        def evaluate(self):
            """
            Runs the evaluation pipeline.
            """
            pass