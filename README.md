# GlyBot
A Domain-Specific LLM Assistant for Glycans and Glycobiology

### *WIP*

**Overview**

This project aims to develop an AI-assistant capable of querying a bioinformatics knowledgebase and providing assistance navigating and utilizing the informatics tools that are available. Documentation has historically been the primary method of informing users on how to utilize these resources, however, the effort to create, maintain, and utilize said documentation is often disproportionate to its utility. An AI-assistant provides a flexible platform for users with varying levels of experience and domain knowledge to interact with in order to become familiar with a new domain-specific informatics tool.

**Abstract**

Many bioinformatics tools require a significant level of domain-expertise and algorithmic familiarity, making it difficult for new users to fully utilize these resources. Large Language Models (LLMs) have recently emerged as incredibly powerful text-processing tools with multiple applications for advancing scientific research (Buehler 2024), potentially offering a way to bridge this gap with LLM-based AI-assistants to help navigate these tools. Despite their enormous foundational knowledge from data ingested during training, however, these models lack knowledge outside their training corpus and require additional fine-tuning for domain specific applications (Soudani et al 2024). We evaluated the use of Retrieval Augmented Generation (RAG) with semantic vector search for enhancing domain-specific glycobiology knowledge of an LLM-based AI-assistant and found that RAG substantially improves domain-specific information content of LLM responses in a context-dependent manner by retaining phrases and factual claims within retrieved data from a given knowledgebase.

**References**

Varki A, Cummings RD, Esko JD, et al., editors. Essentials of Glycobiology [Internet]. 4th edition. Cold Spring Harbor (NY): Cold Spring Harbor Laboratory Press; 2022. Available from: https://www.ncbi.nlm.nih.gov/books/NBK579918/ doi: 10.1101/9781621824213 

Yuanjie Lyu and Zhiyu Li and Simin Niu and Feiyu Xiong and Bo Tang and Wenjin Wang and Hao Wu and Huanyong Liu and Tong Xu and Enhong Chen and Yi Luo and Peng Cheng and Haiying Deng and Zhonghao Wang and Zijia Lu. 2024. CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models. arXiv. https://doi.org/10.48550/arXiv.2401.17043 

Rickard Stureborg, Dimitris Alikaniotis, Yoshi Suhara. 2024. Large language models are inconsistent and biased evaluators. arXiv:2405.01724. 2 May 2024. https://doi.org/10.48550/arXiv.2405.01724  

Soudani, H., Kanoulas, E., & Hasibi, F. (2024). Fine Tuning vs. Retrieval Augmented Generation for Less Popular Knowledge. ArXiv, abs/2403.01432. 

Markus J. Buehler. 2024. Accelerating scientific discovery with generative knowledge extraction, graph-based representation, and multimodal intelligent graph reasoning. arXiv:2403.11996v3. 10 Jun 2024.

Gao Silin. 2024. Efficient Tool Use with Chain-of-Abstraction Reasoning. arXiv:2401.17464. https://doi.org/10.48550/arXiv.2401.17464

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. BLEU: a Method for Automatic Evaluation of Machine Translation. Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (ACL), Philadelphia, July 2002, pp. 311-318.
