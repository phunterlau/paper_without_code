import os
import random
import re
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simulated knowledge graph
knowledge_graph = {
    "products": {
        "laptop123": {"name": "TechPro Laptop", "price": 999, "category": "Electronics", "battery_life": "8 hours"},
        "phone456": {"name": "SmartPhone X", "price": 699, "category": "Electronics", "battery_life": "12 hours"},
    },
    "customer_orders": {
        "789": {"customer": "John Doe", "product": "laptop123", "status": "Shipped"},
    },
    "reviews": {
        "laptop123": [{"rating": 4.5, "text": "Great performance for the price!"}],
        "phone456": [{"rating": 4.0, "text": "Good phone, but battery life could be better."}],
    }
}

# Simulated retrieval tools
class KnowledgeGraphRetriever:
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph

    def _get_product_id(self, product_name):
        for product_id, product_info in self.knowledge_graph["products"].items():
            if product_info["name"].lower() == product_name.lower():
                return product_id
        return None

    def prod_qna(self, product_id_or_name, query):
        product_id = self._get_product_id(product_id_or_name) or product_id_or_name
        product = self.knowledge_graph["products"].get(product_id)
        if product:
            query = query.lower()
            if query == "price":
                return f"The price of {product['name']} is ${product['price']}."
            elif query == "battery life":
                return f"The battery life of {product['name']} is approximately {product.get('battery_life', '12 hours')}."
            elif query == "availability":
                return f"{product['name']} is currently {'in stock' if random.choice([True, False]) else 'out of stock'}."
            elif query == "return policy":
                return f"The return policy for {product['name']} is 30 days."
            elif query == "warranty":
                return f"The warranty for {product['name']} is {product.get('warranty', '1 year')}."
            else:
                return f"Product: {product['name']}, Price: ${product['price']}, Category: {product['category']}"
        return "Product not found"

    def order_status(self, order_id):
        order = self.knowledge_graph["customer_orders"].get(order_id)
        if order:
            return f"Order {order_id} for {order['product']} is {order['status']}"
        return "Order not found"

    def review_summary(self, product_id_or_name):
        product_id = self._get_product_id(product_id_or_name) or product_id_or_name
        reviews = self.knowledge_graph["reviews"].get(product_id, [])
        if reviews:
            avg_rating = sum(review['rating'] for review in reviews) / len(reviews)
            return f"Average rating for {product_id_or_name}: {avg_rating:.1f}, Sample review: {reviews[0]['text']}"
        return "No reviews found"

    # Alias methods to handle variations
    get_prod_qna = prod_qna_v2 = prod_qna
    get_review_summary = review_summary_v2 = review_summary
    order_status_v2 = order_status

retriever = KnowledgeGraphRetriever(knowledge_graph)

# Tool Evolve (TEvo)
# This function creates variations of tool descriptions to enhance model robustness
def tevo(tools):
    evolved_tools = []
    for tool in tools:
        name_variations = [tool['name'], f"{tool['name']}_v2", f"get_{tool['name']}"]
        desc_variations = [
            tool['description'],
            f"This tool {tool['description'].lower()}",
            f"Use this to {tool['description'].lower()}"
        ]
        evolved_tools.append({
            'name': random.choice(name_variations),
            'description': random.choice(desc_variations)
        })
    return evolved_tools

# Tool-Task Generator (TTG)
# This function generates diverse, retrieval-related tasks to improve the model's reasoning capabilities
def ttg(primary_task):
    secondary_tasks = [
        f"Explain why the tool in Step 1 is the most appropriate for answering the query.",
        "What additional information might be needed to provide a more comprehensive answer?",
        "If the primary tool fails, what would be an alternative approach to answer the query?",
        "How could this plan be expanded to provide more detailed information to the customer?",
        "Identify any assumptions made in this plan and explain how they might affect the response.",
    ]
    return random.choice(secondary_tasks)

# Diverse Query Sampler (DQS)
# This function selects a diverse set of queries based on their semantic similarity
def get_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

def dqs(initial_queries, additional_queries, n_samples=5):
    initial_embeddings = [get_embedding(q) for q in initial_queries]
    additional_embeddings = [get_embedding(q) for q in additional_queries]
    
    similarities = cosine_similarity(initial_embeddings, additional_embeddings)
    diverse_indices = np.argsort(similarities.max(axis=0))[:n_samples]
    
    return [additional_queries[i] for i in diverse_indices]

# REAPER prompt
# This function generates the prompt for the REAPER model, incorporating evolved tool descriptions
def get_reaper_prompt(tools):
    evolved_tools = tevo(tools)
    prompt = """You are an AI assistant for a retail store. Generate a single, coherent step-by-step plan to retrieve information from the knowledge graph to answer customer questions. Use these tools:

"""
    for tool in evolved_tools:
        prompt += f"{tool['name']}: {tool['description']}\n"
    prompt += """
Generate a plan using these tools. Each step should use only one tool and should be in the following format:
Step X: tool_name("arg1", "arg2")

If no tool is needed, state that the question can be answered without retrieval. Do not generate multiple plans.

Customer query: {query}
Context: {context}

Plan:
"""
    return prompt

# REAPER function
# This is the main function that generates the retrieval plan
def reaper(query, context="", model="gpt-4o"):
    tools = [
        {'name': 'prod_qna', 'description': 'Fetches specific information for a product.'},
        {'name': 'order_status', 'description': 'Retrieves the status of an order.'},
        {'name': 'review_summary', 'description': 'Provides a summary of reviews for a product.'}
    ]
    prompt = get_reaper_prompt(tools).format(query=query, context=context)
    
    # Apply TTG to create a secondary task
    secondary_task = ttg(query)
    prompt += f"\n\nAdditional task: {secondary_task}"
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    
    return response.choices[0].message.content, secondary_task

# Execute plan function
# This function executes the plan generated by REAPER
def execute_plan(plan):
    results = []
    steps = [step.strip() for step in plan.split('\n') if step.strip().startswith('Step')]
    
    for step in steps:
        match = re.search(r'Step \d+:\s*(\w+)\s*\((.*?)\)', step)
        if match:
            tool_name, args = match.groups()
            args = [arg.strip().strip('"') for arg in args.split(',')]
            
            if tool_name in ['prod_qna', 'prod_qna_v2', 'get_prod_qna']:
                if len(args) == 2:
                    results.append(retriever.prod_qna(args[0], args[1]))
                else:
                    results.append(f"Error: Invalid number of arguments for {tool_name}")
            elif tool_name in ['review_summary', 'get_review_summary', 'review_summary_v2']:
                if len(args) == 1:
                    results.append(retriever.review_summary(args[0]))
                else:
                    results.append(f"Error: Invalid number of arguments for {tool_name}")
            elif tool_name in ['order_status', 'order_status_v2']:
                if len(args) == 1:
                    results.append(retriever.order_status(args[0]))
                else:
                    results.append(f"Error: Invalid number of arguments for {tool_name}")
            else:
                results.append(f"Unknown tool: {tool_name}")
    
    return "\n".join(results) if results else "Could not execute any steps in the plan."

# Main execution
if __name__ == "__main__":
    # Initial set of queries
    initial_queries = [
        "What's the price of the TechPro Laptop?",
        "What's the status of my order number 789?",
    ]

    # Additional set of queries
    additional_queries = [
        "What's the battery life of the SmartPhone X?",
        "Can you summarize the reviews for the SmartPhone X?",
        "Is the TechPro Laptop available, and what's its return policy?",
        "What accessories are available for the SmartPhone X and how much do they cost?",
        "Compare the battery life of the SmartPhone X and the TechPro Laptop",
        "What's the warranty and price of the SmartPhone X?",
        "Are there any negative reviews about the battery life of the SmartPhone X?",
        "What's the status of my order 123 and when will it be delivered?",
        "Tell me about the camera quality and storage options for the SmartPhone X",
        "What's the most popular feature of the TechPro Laptop according to customer reviews?",
    ]

    print("Initial Queries:")
    for query in initial_queries:
        print(f"- {query}")
    print("\nAdditional Queries:")
    for query in additional_queries:
        print(f"- {query}")

    # Use DQS to select diverse queries
    diverse_queries = dqs(initial_queries, additional_queries)

    print("\nDiverse Queries selected by DQS:")
    for query in diverse_queries:
        print(f"- {query}")

    print("\nExecuting REAPER for Initial and Diverse Queries:")
    all_queries = initial_queries + diverse_queries

    for query in all_queries:
        print(f"\nQuery: {query}")
        plan, secondary_task = reaper(query)
        
        # Extracting main plan
        main_plan = plan.split("Plan:")[-1].strip() if "Plan:" in plan else plan

        print("REAPER Plan:")
        print(main_plan)
        print("\nAdditional Task:")
        print(secondary_task)
        print("\nExecution Result:")
        result = execute_plan(main_plan)
        print(result)
        print("=" * 70)