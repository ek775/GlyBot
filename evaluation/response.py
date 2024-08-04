"""Script for running Create Read Update Delete Metrics from API/Model Responses"""

# imports
import pandas as pd
import os

from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from ragas.metrics import (
    # response metrics
    faithfulness,
    answer_relevancy,
    answer_similarity,
    # retrieval metrics
    context_relevancy,
    context_precision,
    context_recall,
)
from ragas.integrations.llama_index import evaluate

class GlyBot_Evaluator():
    """
    Class for running response evaluation in the main experimentation pipeline.
    """
    def __init__(self, curated_q_path: str, query_engine: RetrieverQueryEngine):
        """
        Initialize the evaluator with the test data and query engine.
        """
        self.curated_q_path = curated_q_path
        self.query_engine = query_engine
        self.ragas_metrics = [
            # response metrics
            faithfulness,
            answer_relevancy,
            answer_similarity,
            # retrieval metrics
            context_relevancy,
            context_precision,
            context_recall,
            ]   
        self.curated_dict = None

    def set_query_engine(self, query_engine):
        """
        Set the query engine for the evaluator.
        """
        self.query_engine = query_engine

    def get_prompts(self):
        """
        Processes the test data for evaluation.
        """

        # load curated test set
        curated_test_set = pd.read_csv(self.curated_q_path)
        columns = curated_test_set.columns
        curated_dict = {"question":curated_test_set[columns[1]],
                        "ground_truth":curated_test_set[columns[2]]}

        self.curated_dict = curated_dict

    def response_evaluation(self, metadata: dict) -> None:
        """
        Runs the evaluation pipeline. Results are stored as .csv files in a new directory.
        """
        llm = Settings.llm
        embed_model = Settings.embed_model

        ragas_curated = evaluate(
            query_engine = self.query_engine,
            metrics = self.ragas_metrics,
            dataset = self.curated_dict,
            llm=llm,
            embeddings=embed_model,
            raise_exceptions=False
        )
        results = [ 
                   ("ragas_curated",ragas_curated)
                   ]
        
        def set_result_path():
            """
            recursively tries to set a new result path for each evaluation run
            so results are not saved over each other.
            """
            result_path = "./results/response_evaluation"
            loc = 0
            while os.path.exists(result_path):
                loc+=1
                result_path = f"./results/response_evaluation_{loc}"

            return result_path

        path = set_result_path() 
        os.system(f"mkdir {path}") 

        # save metadata
        with open(f"{path}/metadata.txt", 'w') as f:
            f.write(str(metadata))
        # save results  
        for data in results:
            df = data[1].to_pandas()
            df.to_csv(f"{path}/{data[0]}.csv", encoding='utf-8', index=False)