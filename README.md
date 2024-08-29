# GlyBot
A Domain-Specific LLM Assistant for Glycans and Glycobiology

#

**Overview**

The Assistant is currently in a limited prototype stage on Streamlit Community Cloud here: [GlyBot](https://glybot.streamlit.app/)

This project aims to develop an AI-assistant capable of querying a bioinformatics knowledgebase and providing assistance navigating and utilizing the informatics tools that are available. An AI-assistant provides a flexible platform for users with varying ranges of experience and domain knowledge to interact with in order to become familiar with a new domain-specific informatics tool. 

The initial feasability step of this project involved building and evaluating a RAG pipeline to understand the principles of prompt engineering, RAG, and its effects on LLM responses. Data and analysis from this part of the project can be found in the results directory of this repository.

Current efforts are aimed at enhancing the Assistant with various API tools for retrieving information from GlyGen, UniProt, and other relevant resources for glycoproteomics and achieving coherent integration of this information for assisting researchers. 

If you would like to contribute to this project, reach out to me via email at ek990@georgetown.edu

**Run with Docker**

To use the prototype assistant yourself, the easist method is via docker container. Due to the use of external APIs, these instructions use docker files in the repo to build the image using your API keys and assume that you have installed and/or are familiar with git and docker command line tools. 

Steps:
1. Clone the repository

```
# locally
git clone https://github.com/ek775/GlyBot.git
# with github CLI
gh repo clone ek775/GlyBot
```

2. Add your API keys

Once you have done this, you will need to supply api keys for OpenAI and Google, which, you can get here:

[OpenAI Developer Portal](https://platform.openai.com) | [Google Custom Search](https://console.cloud.google.com/apis/library/customsearch.googleapis.com)

Put these into a folder labelled SENSITIVE as text files,
> GlyBot/SENSITIVE/openai_api_key.txt

> GlyBot/SENSITIVE/google_api_key.txt

3. Build the Docker image

```
docker build . -t glybot:myassistant
```

4. Run the container and open in browser

```
docker container run -d -p 8501:8501 glybot:myassistant
```

Access the application by going to > http://localhost:8501/

**Python Guide**

To interact with assistant locally, clone the repository and install the required python packages. This can be done using either anaconda or pip using the following commands:

*Anaconda*

> conda env create -f conda_dev_env.yml
> conda activate glybot

*Pip*
> pip install $(< requirements.txt)

Once you have done this, you will need to supply api keys for OpenAI and Google, which, you can get here:

[OpenAI Developer Portal](https://platform.openai.com) | [Google Custom Search](https://console.cloud.google.com/apis/library/customsearch.googleapis.com)

Put these into a folder labelled SENSITIVE as text files,
> openai_api_key.txt

> google_api_key.txt

or set them as environment variables and run a local streamlit server with:
> streamlit run streamlit_app.py

To run in your terminal or replicate my analysis, you can run the main script with:
> python ./ [openai/ollama] [chat/eval]

#

**References**

Varki A, Cummings RD, Esko JD, et al., editors. Essentials of Glycobiology [Internet]. 4th edition. Cold Spring Harbor (NY): Cold Spring Harbor Laboratory Press; 2022. Available from: https://www.ncbi.nlm.nih.gov/books/NBK579918/ doi: 10.1101/9781621824213 

Yuanjie Lyu and Zhiyu Li and Simin Niu and Feiyu Xiong and Bo Tang and Wenjin Wang and Hao Wu and Huanyong Liu and Tong Xu and Enhong Chen and Yi Luo and Peng Cheng and Haiying Deng and Zhonghao Wang and Zijia Lu. 2024. CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models. arXiv. https://doi.org/10.48550/arXiv.2401.17043 

Rickard Stureborg, Dimitris Alikaniotis, Yoshi Suhara. 2024. Large language models are inconsistent and biased evaluators. arXiv:2405.01724. 2 May 2024. https://doi.org/10.48550/arXiv.2405.01724  

Soudani, H., Kanoulas, E., & Hasibi, F. (2024). Fine Tuning vs. Retrieval Augmented Generation for Less Popular Knowledge. ArXiv, abs/2403.01432. 

Markus J. Buehler. 2024. Accelerating scientific discovery with generative knowledge extraction, graph-based representation, and multimodal intelligent graph reasoning. arXiv:2403.11996v3. 10 Jun 2024.

Gao Silin. 2024. Efficient Tool Use with Chain-of-Abstraction Reasoning. arXiv:2401.17464. https://doi.org/10.48550/arXiv.2401.17464

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. BLEU: a Method for Automatic Evaluation of Machine Translation. Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (ACL), Philadelphia, July 2002, pp. 311-318.
