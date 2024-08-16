Paper link "HybridRAG: Integrating Knowledge Graphs and Vector Retrieval Augmented Generation for Efficient Information Extraction" <https://arxiv.org/abs/2408.04948>

"""
Workflow of the HybridRAG system:

1. Data Preparation:
   - Documents are prepared with associated timestamps.
   - A knowledge graph is constructed with timestamped triplets.

2. Initialization:
   - VectorRAG is initialized with documents and their embeddings.
   - GraphRAG is initialized with the knowledge graph.
   - HybridRAG is set up to use both VectorRAG and GraphRAG.

3. Query Processing:
   - When a query is received, HybridRAG initiates the retrieval process.

4. Retrieval:
   - VectorRAG retrieves relevant documents based on embedding similarity and recency.
   - GraphRAG retrieves relevant subgraphs from the knowledge graph based on entities in the query.
   - HybridRAG combines the contexts from both sources.

5. Answer Generation:
   - The combined context is used to create a prompt for GPT-4.
   - GPT-4 generates an answer based on the provided context and query.

6. Output:
   - The generated answer is returned to the user.

This implementation demonstrates the core concepts of the HybridRAG approach:
- Combining vector-based and graph-based retrieval for more comprehensive context gathering.
- Incorporating temporal information to prioritize recent data.
- Using a large language model (GPT-4) for answer generation based on the retrieved context.

Areas for potential improvement:
- Implementing a multi-step reasoning process as mentioned in the original paper.
- Enhancing the knowledge graph with more complex relationships and better entity recognition.
- Improving the context integration method in the HybridRAG class.
- Implementing evaluation metrics as described in the paper (faithfulness, answer relevance, context precision, context recall).
"""
