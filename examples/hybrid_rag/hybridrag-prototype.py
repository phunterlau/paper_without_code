import os
import openai
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import dspy
from datetime import datetime

# Set up OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.Client()

# Knowledge Graph setup
class KnowledgeGraph:
    """
    Represents a simple knowledge graph structure.
    This is part of the GraphRAG component, storing information as triplets (head, relation, tail).
    """
    def __init__(self):
        self.graph = nx.Graph()

    def add_triplet(self, head, relation, tail, timestamp=None):
        """
        Adds a triplet to the knowledge graph with an optional timestamp.
        Timestamps allow for prioritizing more recent information during retrieval.
        """
        self.graph.add_edge(head, tail, relation=relation, timestamp=timestamp)

    def get_subgraph(self, entity, depth=1):
        """
        Retrieves a subgraph centered around a given entity.
        This is used in the GraphRAG retrieval process to find relevant information.
        """
        subgraph = nx.ego_graph(self.graph, entity, radius=depth)
        return [(u, v, self.graph[u][v]['relation'], self.graph[u][v].get('timestamp')) for u, v in subgraph.edges()]

# VectorRAG implementation
class VectorRAG:
    """
    Implements the vector-based retrieval component of HybridRAG.
    This class handles document embedding and similarity-based retrieval.
    """
    def __init__(self, documents, timestamps):
        self.documents = documents
        self.timestamps = timestamps
        self.embeddings = self.compute_embeddings(documents)

    def compute_embeddings(self, texts):
        """
        Computes embeddings for the given texts using OpenAI's embedding model.
        These embeddings are used for similarity-based retrieval.
        """
        response = client.embeddings.create(input=texts, model="text-embedding-3-small")
        return [embedding.embedding for embedding in response.data]

    def retrieve(self, query, k=3):
        """
        Retrieves the top-k most relevant documents based on cosine similarity.
        The retrieval is adjusted based on the recency of the documents.
        """
        query_embedding = self.compute_embeddings([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Adjust similarities based on recency
        current_time = datetime.now()
        time_weights = [(current_time - timestamp).days for timestamp in self.timestamps]
        max_weight = max(time_weights)
        normalized_weights = [1 - (w / max_weight) for w in time_weights]
        
        adjusted_similarities = similarities * normalized_weights
        
        top_k_indices = np.argsort(adjusted_similarities)[-k:][::-1]
        return [self.documents[i] for i in top_k_indices]

# GraphRAG implementation
class GraphRAG:
    """
    Implements the graph-based retrieval component of HybridRAG.
    This class uses the knowledge graph to retrieve relevant information.
    """
    def __init__(self, kg):
        self.kg = kg

    def retrieve(self, query):
        """
        Retrieves relevant information from the knowledge graph based on entities in the query.
        This method finds entities in the query and retrieves related subgraphs.
        """
        entities = [word for word in query.split() if word in self.kg.graph.nodes()]
        if not entities:
            return []
        
        entity = entities[0]
        return self.kg.get_subgraph(entity)

# HybridRAG implementation
class HybridRAG:
    """
    Implements the HybridRAG approach, combining VectorRAG and GraphRAG.
    This class orchestrates the retrieval from both sources and generates the final answer.
    """
    def __init__(self, vectorrag, graphrag):
        self.vectorrag = vectorrag
        self.graphrag = graphrag

    def retrieve(self, query):
        """
        Retrieves context from both VectorRAG and GraphRAG.
        This combined approach aims to leverage the strengths of both retrieval methods.
        """
        vector_context = self.vectorrag.retrieve(query)
        graph_context = self.graphrag.retrieve(query)
        return vector_context + [f"{head} {relation} {tail}" for head, tail, relation, _ in graph_context]

    def generate_answer(self, query):
        """
        Generates an answer to the query using the retrieved context.
        This method uses GPT-4 to produce a response based on the combined context from VectorRAG and GraphRAG.
        """
        context = self.retrieve(query)
        prompt = f"""Based on the following context, answer the question. Be specific and use the information provided in the context.

Context:
{' '.join(context)}

Question: {query}

Answer:"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that answers questions based on the given context. Always use the information from the context if it's relevant to the question."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

# Sample data with timestamps
# This data simulates a small corpus of information about Apple Inc.
documents = [
    "Apple Inc. is a technology company headquartered in Cupertino, California.",
    "Tim Cook is the CEO of Apple Inc. since 2011.",
    "Apple's latest iPhone model features advanced AI capabilities.",
    "The company reported strong financial results in Q4 2023, with revenue growth of 8% year-over-year."
]

timestamps = [
    datetime(2022, 1, 1),
    datetime(2022, 6, 1),
    datetime(2023, 9, 1),
    datetime(2023, 12, 1)
]

# Create and populate the knowledge graph
kg = KnowledgeGraph()
kg.add_triplet("Apple Inc.", "is_a", "technology company", datetime(2022, 1, 1))
kg.add_triplet("Apple Inc.", "headquartered_in", "Cupertino", datetime(2022, 1, 1))
kg.add_triplet("Tim Cook", "is_CEO_of", "Apple Inc.", datetime(2022, 6, 1))
kg.add_triplet("iPhone", "is_product_of", "Apple Inc.", datetime(2023, 9, 1))
kg.add_triplet("Apple Inc.", "reported", "strong financial results", datetime(2023, 12, 1))
kg.add_triplet("Apple Inc.", "achieved", "8% revenue growth", datetime(2023, 12, 1))

# Set up RAG systems
vectorrag = VectorRAG(documents, timestamps)
graphrag = GraphRAG(kg)
hybridrag = HybridRAG(vectorrag, graphrag)

# Demonstration
query = "What can you tell me about Apple's recent performance?"
answer = hybridrag.generate_answer(query)
print(f"Query: {query}")
print(f"Answer: {answer}")

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