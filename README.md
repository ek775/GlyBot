# GlyBot
A Domain-Specific LLM Assistant for Glycans and Glycobiology

#

**Overview**

This project aims to develop an AI-assistant capable of querying a bioinformatics knowledgebase and providing assistance navigating and utilizing the informatics tools that are available. An AI-assistant provides a flexible platform for users with varying ranges of experience and domain knowledge to interact with in order to become familiar with a new domain-specific informatics tool. 

*The assistant is currently in a prototype stage* and current efforts are aimed at enhancing the assistant with various API tools for retrieving information from GlyGen, UniProt, and other relevant resources for glycoproteomics and achieving coherent integration of this information for assisting researchers. 

If you would like to contribute to this project, reach out to me via email at ek990@georgetown.edu

#

**Run with Docker**

To experiment with the prototype assistant yourself, this repository is configured to use docker compose to run the application and vector database as a pair of networked containers. You will need [docker compose](https://docs.docker.com/compose/) installed appropriately for your machine, then follow the steps below.

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

Access the application by going to http://localhost:8501/

# Initial Feasability Analysis

Early work to evaluate the efficacy of RAG and its effects on LLM output in this domain can be found in its own branch of this repository. If you are interested in that work, it can be found in the RAG_feasability_eval branch.

#

**References**

Varki A, Cummings RD, Esko JD, et al., editors. Essentials of Glycobiology [Internet]. 4th edition. Cold Spring Harbor (NY): Cold Spring Harbor Laboratory Press; 2022. Available from: https://www.ncbi.nlm.nih.gov/books/NBK579918/ doi: 10.1101/9781621824213 

Yuanjie Lyu and Zhiyu Li and Simin Niu and Feiyu Xiong and Bo Tang and Wenjin Wang and Hao Wu and Huanyong Liu and Tong Xu and Enhong Chen and Yi Luo and Peng Cheng and Haiying Deng and Zhonghao Wang and Zijia Lu. 2024. CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models. arXiv. https://doi.org/10.48550/arXiv.2401.17043 

Rickard Stureborg, Dimitris Alikaniotis, Yoshi Suhara. 2024. Large language models are inconsistent and biased evaluators. arXiv:2405.01724. 2 May 2024. https://doi.org/10.48550/arXiv.2405.01724  

Soudani, H., Kanoulas, E., & Hasibi, F. (2024). Fine Tuning vs. Retrieval Augmented Generation for Less Popular Knowledge. ArXiv, abs/2403.01432. 

Markus J. Buehler. 2024. Accelerating scientific discovery with generative knowledge extraction, graph-based representation, and multimodal intelligent graph reasoning. arXiv:2403.11996v3. 10 Jun 2024.

Gao Silin. 2024. Efficient Tool Use with Chain-of-Abstraction Reasoning. arXiv:2401.17464. https://doi.org/10.48550/arXiv.2401.17464

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. BLEU: a Method for Automatic Evaluation of Machine Translation. Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (ACL), Philadelphia, July 2002, pp. 311-318.
