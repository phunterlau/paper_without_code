import os
import openai
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# Set up OpenAI client
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.Client()

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
        This method implements a three-pass approach as described in the paper:
        1. First pass: General understanding
        2. Second pass: Detailed analysis
        3. Third pass: Final answer generation
        """
        context = self.retrieve(query)
        
        # First pass: General understanding
        first_pass_prompt = f"""Based on the following context, provide a general understanding of the topic related to the question. Don't answer the question directly yet.

Context:
{' '.join(context)}

Question: {query}

General understanding:"""
        
        first_pass_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant providing a general understanding of a topic based on given context."},
                {"role": "user", "content": first_pass_prompt}
            ]
        )
        first_pass_understanding = first_pass_response.choices[0].message.content

        # Second pass: Detailed analysis
        second_pass_prompt = f"""Using the general understanding and the original context, perform a detailed analysis to answer the question. Focus on specific details and relationships.

General understanding:
{first_pass_understanding}

Original context:
{' '.join(context)}

Question: {query}

Detailed analysis:"""

        second_pass_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant performing a detailed analysis to answer a question based on given context and understanding."},
                {"role": "user", "content": second_pass_prompt}
            ]
        )
        detailed_analysis = second_pass_response.choices[0].message.content

        # Third pass: Final answer generation
        final_prompt = f"""Based on the general understanding and detailed analysis, provide a concise and accurate answer to the question.

General understanding:
{first_pass_understanding}

Detailed analysis:
{detailed_analysis}

Question: {query}

Final answer:"""

        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant providing a final, concise answer based on previous analysis."},
                {"role": "user", "content": final_prompt}
            ]
        )
        return final_response.choices[0].message.content

# Sample data with timestamps
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

5. Multi-Pass Answer Generation:
   a. First Pass: General Understanding
      - Provides an overall understanding of the topic based on the retrieved context.
   b. Second Pass: Detailed Analysis
      - Performs a more in-depth analysis using the general understanding and original context.
   c. Third Pass: Final Answer Generation
      - Synthesizes the previous analyses to generate a concise and accurate answer.

6. Output:
   - The generated answer is returned to the user.

This implementation demonstrates the core concepts of the HybridRAG approach:
- Combining vector-based and graph-based retrieval for more comprehensive context gathering.
- Incorporating temporal information to prioritize recent data.
- Using a multi-pass approach for answer generation to improve comprehension and accuracy.

Areas for potential improvement:
- Implementing more sophisticated graph traversal algorithms.
- Enhancing the integration method of vector and graph-based retrieval results.
- Implementing evaluation metrics as described in the paper (faithfulness, answer relevance, context precision, context recall).
- Expanding the knowledge graph with more complex relationships and better entity recognition.
"""