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
    def __init__(self, curated_q_path, documents, query_engine):
        """
        Initialize the evaluator with the test data and query engine.
        """
        self.documents = documents
        self.curated_q_path = curated_q_path
        self.query_engine = query_engine
        self.llm_metrics = [
            FaithfulnessEvaluator(),
            RelevancyEvaluator(),
            CorrectnessEvaluator()
        ]
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
            # generate synthetic test set
            docs = self.documents
            generator = TestsetGenerator.from_llama_index()
            synthetic_test_set = generator.generate_with_llamaindex_docs(
                documents = docs,
                test_size=20,
                distributions={
                    simple: 0.5, 
                    reasoning: 0.25, 
                    multi_context: 0.25}
                )
            synthetic_dict = synthetic_test_set.to_dataset().to_dict()
            self.synthetic_dict = synthetic_dict

            # load curated test set
            curated_test_set = pd.read_csv(self.curated_q_path)
            curated_dict = {"question":curated_test_set[],
                            "ground_truth":curated_test_set[]}

            self.curated_dict = None

        def response_evaluation(self):
            """
            Runs the evaluation pipeline.
            """
            ragas_synth = evaluate(
                query_engine = self.query_engine,
                metrics = self.ragas_metrics
                dataset = self.synthetic_dict
            )
            llm_synth = evaluate(
                query_engine = self.query_engine,
                metrics = self.llm_metrics
                dataset = self.synthetic_dict
            )
            ragas_curated = evaluate(
                query_engine = self.query_engine,
                metrics = self.ragas_metrics
                dataset = self.curated_dict
            )
            llm_curated = evaluate(
                query_engine = self.query_engine,
                metrics = self.llm_metrics
                dataset = self.curated_dict
            )
            results = [("ragas_synth",ragas_synth), 
                       ("llm_synth",llm_synth), 
                       ("ragas_curated",ragas_curated), 
                       ("llm_curated",llm_curated)]
            for data in results:
                df = data[1].to_pandas()
                df.to_csv(f"./response_evaluation/{data[0]}.csv")