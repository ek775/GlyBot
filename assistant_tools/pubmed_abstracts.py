from llama_index.readers.papers import PubmedReader
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field


### pubmed reader tool
def build_pubmed_search_tool() -> FunctionTool:
    print("Building PubMed Search Tool")
    class PubMedQuery(BaseModel):
        """pydantic object describing how to search pubmed to the llm"""
        query: str = Field(..., description="Natural Language Query to search for on Pubmed.")

    def pubmed_search(query: str):
        """Retrieves abstracts of relevant papers from PubMed"""
        print("PubMed Search query:",query)
        reader = PubmedReader()
        papers = reader.load_data(search_query=query, max_results=10)
        print("papers:")
        print("----------------")
        print(papers[0].text)
        print("----------------")
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
