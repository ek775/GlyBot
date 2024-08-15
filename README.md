# GlyBot
A Domain-Specific LLM Assistant for Glycans and Glycobiology

### *WIP*

#

**Overview**

The Assistant is currently in a limited prototype stage on Streamlit Community Cloud here: [GlyBot](https://glybot.streamlit.app/)

This project aims to develop an AI-assistant capable of querying a bioinformatics knowledgebase and providing assistance navigating and utilizing the informatics tools that are available. An AI-assistant provides a flexible platform for users with varying ranges of experience and domain knowledge to interact with in order to become familiar with a new domain-specific informatics tool. 

The initial feasability step of this project involved building and evaluating a RAG pipeline to understand the principles of prompt engineering, RAG, and its effects on LLM responses. Data and analysis from this part of the project can be found in the results directory of this repository.

Current efforts are aimed at enhancing the Assistant with various API tools for retrieving information from GlyGen, UniProt, and other relevant resources for glycoproteomics and achieving coherent integration of this information for assisting researchers. 

If you would like to contribute to this project, reach out to me via email at ek990@georgetown.edu

**Repository Guide**



#

**Abstract**

Many bioinformatics tools are challenging for new users because they require deep expertise and familiarity with complex algorithms. However, recent advancements in Large Language Models (LLMs) have shown that these powerful text-processing tools can assist with various scientific research tasks (Buehler 2024), potentially making it easier for non-experts to use these tools. Although LLMs are built on a vast amount of information, they have limitations when it comes to knowledge not included in their initial training, especially in specialized fields (Soudani et al 2024). To address this, we explored the use of a method called Retrieval Augmented Generation (RAG), combined with semantic vector search, to enhance the glycobiology knowledge of an LLM-based AI assistant. This method augments a user's query at execution time with reference information retrieved from a database of pre-processed reliable source material and serves the LLM engineered queries with both questions and reference material to use in generating a response. We compared augmented LLM responses to non-augmented LLM responses on several different tasks using a curated set of questions with human-answers as ground-truth and reference information retrieved from the textbook "Essentials of Glycobiology, 4th edition" (Varki et al 2022) to augment the responses. Our study found that RAG significantly improves the factual content of the AI's responses in this specialized area by incorporating specific phrases and facts from a targeted knowledgebase.

#

**References**

Varki A, Cummings RD, Esko JD, et al., editors. Essentials of Glycobiology [Internet]. 4th edition. Cold Spring Harbor (NY): Cold Spring Harbor Laboratory Press; 2022. Available from: https://www.ncbi.nlm.nih.gov/books/NBK579918/ doi: 10.1101/9781621824213 

Yuanjie Lyu and Zhiyu Li and Simin Niu and Feiyu Xiong and Bo Tang and Wenjin Wang and Hao Wu and Huanyong Liu and Tong Xu and Enhong Chen and Yi Luo and Peng Cheng and Haiying Deng and Zhonghao Wang and Zijia Lu. 2024. CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models. arXiv. https://doi.org/10.48550/arXiv.2401.17043 

Rickard Stureborg, Dimitris Alikaniotis, Yoshi Suhara. 2024. Large language models are inconsistent and biased evaluators. arXiv:2405.01724. 2 May 2024. https://doi.org/10.48550/arXiv.2405.01724  

Soudani, H., Kanoulas, E., & Hasibi, F. (2024). Fine Tuning vs. Retrieval Augmented Generation for Less Popular Knowledge. ArXiv, abs/2403.01432. 

Markus J. Buehler. 2024. Accelerating scientific discovery with generative knowledge extraction, graph-based representation, and multimodal intelligent graph reasoning. arXiv:2403.11996v3. 10 Jun 2024.

Gao Silin. 2024. Efficient Tool Use with Chain-of-Abstraction Reasoning. arXiv:2401.17464. https://doi.org/10.48550/arXiv.2401.17464

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. BLEU: a Method for Automatic Evaluation of Machine Translation. Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (ACL), Philadelphia, July 2002, pp. 311-318.
