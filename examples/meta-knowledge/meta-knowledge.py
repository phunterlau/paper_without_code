import os
import openai
import json
from typing import List, Dict
import numpy as np

# Set up OpenAI client
openai.api_key = os.environ["OPENAI_API_KEY"]
client = openai.Client()

def gpt4_call(system_content: str, user_content: str, json_mode: bool = False) -> dict:
    """
    Make a call to GPT-4 with the given system and user content.
    Force JSON output if json_mode is True.
    """
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format={"type": "json_object"} if json_mode else None
    )
    content = response.choices[0].message.content
    return json.loads(content) if json_mode else content

class DocumentProcessor:
    def process(self, document: str) -> Dict:
        """
        Process a document to generate metadata and synthetic QA pairs.
        This implements the paper's concept of creating substantial, self-contained content.
        """
        system_content = "You are an expert research assistant processing documents."
        user_content = f"""Generate metadata, synthetic questions, and answers for this document:

{document}

Provide metadata including research field, application type, and whether it contains mathematical reasoning. Generate at least 5 synthetic QA pairs.

Output the result as a JSON object with the following structure:
{{
    "metadata": {{
        "research_field": string,
        "application_type": string,
        "contains_math": boolean
    }},
    "qa_pairs": [
        {{
            "question": string,
            "answer": string
        }}
    ]
}}
"""
        result = gpt4_call(system_content, user_content, json_mode=True)
        print("Document Processing Result:")
        print(json.dumps(result, indent=2))
        return result

class MKSummaryGenerator:
    def generate(self, metadata: Dict, questions: List[str]) -> str:
        """
        Generate a Meta Knowledge Summary based on metadata and questions.
        This implements the paper's concept of creating summaries for metadata-based clusters.
        """
        system_content = "You are an expert in summarizing research knowledge."
        user_content = f"""Generate a Meta Knowledge Summary for the following metadata and questions:

Metadata: {json.dumps(metadata)}

Questions: {json.dumps(questions)}

Provide a concise summary that captures the key concepts and areas of focus based on the metadata and questions.
"""
        summary = gpt4_call(system_content, user_content)
        print("Meta Knowledge Summary:")
        print(summary)
        return summary

class QueryAugmenter:
    def augment(self, query: str, mk_summary: str) -> List[str]:
        """
        Augment the user query using the Meta Knowledge Summary.
        This implements the paper's concept of query augmentation for personalized search.
        """
        system_content = "You are an expert in query augmentation for research."
        user_content = f"""Augment this query using the provided Meta Knowledge Summary:

Query: {query}

MK Summary: {mk_summary}

Generate up to 5 augmented queries. Output the result as a JSON object with a single key "augmented_queries" whose value is an array of strings.
"""
        result = gpt4_call(system_content, user_content, json_mode=True)
        augmented_queries = result.get("augmented_queries", [])
        print("Augmented Queries:")
        print(json.dumps(augmented_queries, indent=2))
        return augmented_queries

def retrieve_qa(query: str, qa_pairs: List[Dict[str, str]], k: int = 3) -> List[Dict[str, str]]:
    """
    Retrieve the most relevant QA pairs for a given query.
    This implements the paper's concept of using synthetic questions for retrieval instead of document chunks.
    """
    query_embedding = client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
    question_embeddings = [client.embeddings.create(input=qa['question'], model="text-embedding-3-small").data[0].embedding for qa in qa_pairs]
    
    similarities = [np.dot(query_embedding, qe) for qe in question_embeddings]
    top_k_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:k]
    
    retrieved_qa = [qa_pairs[i] for i in top_k_indices]
    print(f"Retrieved {len(retrieved_qa)} QA pairs for query: {query}")
    return retrieved_qa

class RAGPipeline:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.mk_summary_generator = MKSummaryGenerator()
        self.query_augmenter = QueryAugmenter()

    def process_document(self, document: str) -> Dict:
        """
        Process a document to generate metadata and synthetic QA pairs.
        This is part of the "prepare" step in the paper's prepare-then-rewrite-then-retrieve-then-read workflow.
        """
        return self.document_processor.process(document)

    def generate_mk_summary(self, metadata: Dict, questions: List[str]) -> str:
        """
        Generate a Meta Knowledge Summary.
        This implements the paper's concept of creating summaries for metadata-based clusters.
        """
        return self.mk_summary_generator.generate(metadata, questions)

    def augment_query(self, query: str, mk_summary: str) -> List[str]:
        """
        Augment the user query.
        This is part of the "rewrite" step in the paper's workflow.
        """
        return self.query_augmenter.augment(query, mk_summary)

    def retrieve_and_generate(self, query: str, mk_summary: str, qa_pairs: List[Dict[str, str]]) -> str:
        """
        Retrieve relevant QA pairs and generate a final response.
        This implements the "retrieve" and "read" steps in the paper's workflow.
        """
        augmented_queries = self.augment_query(query, mk_summary)
        all_retrieved_qa = []
        for aug_query in augmented_queries:
            all_retrieved_qa.extend(retrieve_qa(aug_query, qa_pairs))

        context = "\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in all_retrieved_qa])
        
        system_content = "You are a helpful assistant. Use the provided context to answer the user's question."
        user_content = f"""Context:
{context}

Question: {query}

Answer:
"""
        response = gpt4_call(system_content, user_content)
        print("Final Response:")
        print(response)
        return response

def evaluate_response(query: str, response: str) -> Dict[str, int]:
    """
    Evaluate the response using an LLM.
    This implements the paper's concept of using an LLM as a trusted evaluator.
    """
    system_content = "You are an expert evaluator of research responses."
    user_content = f"""Evaluate the following response to the given query on a scale of 0-100 for each metric:

Query: {query}
Response: {response}

Metrics:
1. Recall: Coverage of key, highly relevant information
2. Precision: Ratio of relevant to irrelevant information
3. Specificity: How precisely focused the answer is on the query
4. Breadth: Coverage of all relevant aspects related to the question
5. Depth: Thoroughness of understanding and detailed analysis
6. Relevancy: How well-tailored the answer is to the specific question

Provide your evaluation as a JSON object with metric names as keys and integer scores as values.
"""
    evaluation = gpt4_call(system_content, user_content, json_mode=True)
    print("Evaluation Results:")
    print(json.dumps(evaluation, indent=2))
    return evaluation

# Example usage
if __name__ == "__main__":
    print("Starting RAG Pipeline Demonstration")
    
    document = """
    Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment. 
    The agent receives rewards or penalties based on its actions, aiming to maximize cumulative rewards over time. 
    RL has applications in robotics, game playing, and autonomous systems. Key concepts include the Markov Decision Process, 
    Q-learning, and policy gradients. Recent advancements include Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO).
    """

    print("\nStep 1: Document Processing")
    rag = RAGPipeline()
    result = rag.process_document(document)

    print("\nStep 2: Meta Knowledge Summary Generation")
    mk_summary = rag.generate_mk_summary(result['metadata'], [qa['question'] for qa in result['qa_pairs']])

    print("\nStep 3: Query Processing")
    user_query = "What are the recent advancements in Reinforcement Learning?"
    print(f"User Query: {user_query}")

    print("\nStep 4: Query Augmentation and Response Generation")
    final_response = rag.retrieve_and_generate(user_query, mk_summary, result['qa_pairs'])

    print("\nStep 5: Response Evaluation")
    evaluation = evaluate_response(user_query, final_response)

    print("\nRAG Pipeline Demonstration Completed")