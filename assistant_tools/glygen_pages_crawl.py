from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
import urllib.parse
from llama_index.core import SummaryIndex, Document
import html2text
import selenium.webdriver
import os
import requests
import json

# SSL Issues
import contextlib
import warnings
from urllib3.exceptions import InsecureRequestWarning

old_merge_environment_settings = requests.Session.merge_environment_settings

@contextlib.contextmanager
def no_ssl_verification():
    opened_adapters = set()

    def merge_environment_settings(self, url, proxies, stream, verify, cert):
        # Verification happens only once per connection so we need to close
        # all the opened adapters once we're done. Otherwise, the effects of
        # verify=False persist beyond the end of this context manager.
        opened_adapters.add(self.get_adapter(url))

        settings = old_merge_environment_settings(self, url, proxies, stream, verify, cert)
        settings['verify'] = False

        return settings

    requests.Session.merge_environment_settings = merge_environment_settings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', InsecureRequestWarning)
            yield
    finally:
        requests.Session.merge_environment_settings = old_merge_environment_settings

        for adapter in opened_adapters:
            try:
                adapter.close()
            except:
                pass

### google search tool @glygen
def build_google_search_tool() -> FunctionTool:
    print("Building Google Search Tool")
    # load google api key
    # handled by streamlit secrets
    
    google_custom_search_key = os.environ['GOOGLE_API_KEY']
    google_custom_search_eng = os.environ['GOOGLE_CUSTOM_SEARCH']

    # define tool from function
    class GlyGenGoogleSearch(BaseModel):
        """pydantic object describing how to get information from GlyGen to the llm"""
        query: str = Field(..., description="Natural Language Query to search GlyGen web pages and API docs for.")

    def glygen_google_search(query: str,
                             key: str = google_custom_search_key, 
                             engine: str = google_custom_search_eng,
                             num: int = 5):
        """
        Searches the GlyGen website to find relevant information for navigating the site
        and using the GlyGen APIs to access data.
        """
        # format multiword query
        query_list = query.split()
        url_query = '+'.join(query_list)

        # build the search url
        url_template = ("https://www.googleapis.com/customsearch/v1?key={key}&cx={engine}&q={query}")
        url = url_template.format(key=key, engine=engine, query=urllib.parse.quote_plus(url_query))

        # ensure number of results is not obnoxious (<10)
        if num is not None:
            if not 1 <= num <= 10:
                raise ValueError("num should be an integer between 1 and 10, inclusive")
            url += f"&num={num}"

        # make the initial request
        print("GlyGen glycan page search:",url_query)
        response = requests.get(url)
        results = json.loads(response.content)

        # extract urls to crawl from search results
        pages = results['items']
        documents = []
        with no_ssl_verification():
            chrome_options = selenium.webdriver.ChromeOptions()
            chrome_options.add_argument('--headless=new')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            driver = selenium.webdriver.Chrome(options=chrome_options, keep_alive=True)
            for page in pages:
                driver.get(page['link'])
                md_page = html2text.html2text(driver.page_source)
                documents.append(Document(
                    text=md_page,
                    doc_id=page['title'],
                    metadata=page,
                    )
                )
            driver.quit()

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
