# GlyBot
A Domain-Specific LLM Assistant for Glycans and Glycobiology

### *WIP*

**Overview**

This project aims to develop an AI-assistant capable of querying a bioinformatics knowledgebase and providing assistance navigating and utilizing the informatics tools that are available. Documentation has historically been the primary method of informing users on how to utilize these resources, however, the effort to create, maintain, and utilize said documentation is often disproportionate to its utility. An AI-assistant provides a flexible platform for users with varying levels of experience and domain knowledge to interact with in order to become familiar with a new domain-specific informatics tool.

**Abstract**

Many bioinformatics tools are challenging for new users because they require deep expertise and familiarity with complex algorithms. However, recent advancements in Large Language Models (LLMs) have shown that these powerful text-processing tools can assist with various scientific research tasks, potentially making it easier for non-experts to use these tools. Although LLMs are built on a vast amount of information, they have limitations when it comes to knowledge not included in their initial training, especially in specialized fields. To address this, we explored the use of a method called Retrieval Augmented Generation (RAG), combined with semantic vector search, to enhance the glycobiology knowledge of an LLM-based AI assistant. Our study found that RAG significantly improves the accuracy and relevance of the AI's responses in this specialized area by incorporating specific phrases and facts from a targeted knowledgebase.

**References**

Varki A, Cummings RD, Esko JD, et al., editors. Essentials of Glycobiology [Internet]. 4th edition. Cold Spring Harbor (NY): Cold Spring Harbor Laboratory Press; 2022. Available from: https://www.ncbi.nlm.nih.gov/books/NBK579918/ doi: 10.1101/9781621824213 

Yuanjie Lyu and Zhiyu Li and Simin Niu and Feiyu Xiong and Bo Tang and Wenjin Wang and Hao Wu and Huanyong Liu and Tong Xu and Enhong Chen and Yi Luo and Peng Cheng and Haiying Deng and Zhonghao Wang and Zijia Lu. 2024. CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models. arXiv. https://doi.org/10.48550/arXiv.2401.17043 

Rickard Stureborg, Dimitris Alikaniotis, Yoshi Suhara. 2024. Large language models are inconsistent and biased evaluators. arXiv:2405.01724. 2 May 2024. https://doi.org/10.48550/arXiv.2405.01724  

Soudani, H., Kanoulas, E., & Hasibi, F. (2024). Fine Tuning vs. Retrieval Augmented Generation for Less Popular Knowledge. ArXiv, abs/2403.01432. 

Markus J. Buehler. 2024. Accelerating scientific discovery with generative knowledge extraction, graph-based representation, and multimodal intelligent graph reasoning. arXiv:2403.11996v3. 10 Jun 2024.

Gao Silin. 2024. Efficient Tool Use with Chain-of-Abstraction Reasoning. arXiv:2401.17464. https://doi.org/10.48550/arXiv.2401.17464

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. BLEU: a Method for Automatic Evaluation of Machine Translation. Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics (ACL), Philadelphia, July 2002, pp. 311-318.
