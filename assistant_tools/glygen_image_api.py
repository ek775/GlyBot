from llama_index.core import VectorStoreIndex
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field
import requests
from io import BytesIO
from PIL import Image
import base64

### glygen image search tool
def build_glygen_image_search_tool() -> FunctionTool:
    class GlyGenImageSearch(BaseModel):
        """pydantic object describing how to get glycan images from GlyGen to the llm"""
        query: str = Field(..., 
                           pattern=r"[A-Z]{1}\d{5}[A-Z]{2}",
                           description="Glytoucan accession ID to query for glycan images.")

    def glygen_image_search(query: str):
        """
        Searches GlyGen for images of glycans based on their GlyTouCan accession ID.
        """
        # check if the query is a GlyToucan ID
        if len(query) != 8 or not query[0].isalpha() or not query[1:6].isnumeric() or not query[6:].isalpha():
            return "The query should be a valid GlyTouCan ID"

        # query the api
        url = f"https://api.glygen.org/glycan/image/{query}"
        response = requests.post(url=url, verify=False)

        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img.save("glycan_image_temp_file.png")
            return base64.b64encode(response.content).decode('utf-8')
        elif response.status_code == 404:
            return "No image found for the given GlyTouCan ID"
        else:
            return "An error occurred while fetching the image"
    
    GlyGenImageSearchTool = FunctionTool.from_defaults(
        fn = glygen_image_search,
        name = "glygen_image_search_tool",
        description = "Use this tool to search for glycan images from GlyGen.",
        fn_schema = GlyGenImageSearch)
    
    print("GlyGen Image Search Tool Built")
    return GlyGenImageSearchTool