from llama_index.core import VectorStoreIndex
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
import urllib.parse
from llama_index.core import SummaryIndex
from llama_index.readers.web import SimpleWebPageReader
import os

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