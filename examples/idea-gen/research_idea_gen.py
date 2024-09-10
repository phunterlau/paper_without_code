import os
import json
import requests
from time import sleep
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def search_semantic_scholar(query, limit=10):
    """
    Retrieves relevant papers from Semantic Scholar API.
    This function implements the Retrieval-Augmented Generation (RAG) aspect of the paper.
    """
    base_url = "https://api.semanticscholar.org/graph/v1"
    search_url = f"{base_url}/paper/search"
    
    api_key = os.environ.get("S2_API_KEY")
    headers = {"x-api-key": api_key} if api_key else {}
    
    params = {
        "query": query,
        "fields": "title,abstract,year,citationCount,authors,venue",
        "limit": limit
    }
    
    response = requests.get(search_url, params=params, headers=headers)
    
    if response.status_code != 200:
        print(f"Error: API request failed with status code {response.status_code}")
        return None
    
    return response.json().get('data', [])

def generate_seed_ideas(topic, context, num_seeds=20):
    """
    Generates initial seed ideas.
    This function implements the Seed Idea Generation step from the paper.
    It uses chain-of-thought prompting to encourage more detailed reasoning.
    """
    print(f"Generating {num_seeds} seed ideas for topic: {topic}")
    prompt = f"""Given the following context about the topic "{topic}":

{context}

Generate a brief seed for a novel research idea. The idea should be:
1. Novel and different from existing work
2. Feasible to implement
3. Potentially impactful in the field

Use the following steps:
1. Identify key challenges or gaps in the current research
2. Brainstorm potential solutions or novel approaches
3. Consider interdisciplinary connections
4. Formulate a concise research idea

Provide the seed idea in the following JSON format:
{{
    "title": "Brief title of the idea",
    "description": "One-sentence description of the idea"
}}
"""

    seeds = []
    for i in range(num_seeds):
        print(f"Generating seed idea {i+1}/{num_seeds}")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert AI research assistant capable of generating novel research ideas."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        seed = json.loads(response.choices[0].message.content)
        seeds.append(seed)
        print(f"Generated seed idea:")
        print(f"Title: {seed['title']}")
        print(f"Description: {seed['description']}")
        print()
    return seeds

def deduplicate_ideas(ideas, similarity_threshold=0.8):
    """
    Removes duplicate or very similar ideas.
    This function implements the deduplication step mentioned in the paper,
    using embedding similarity to identify and remove near-duplicate ideas.
    """
    print(f"Deduplicating {len(ideas)} ideas...")
    def get_embedding(text):
        response = client.embeddings.create(input=text, model="text-embedding-ada-002")
        return response.data[0].embedding

    embeddings = [get_embedding(idea) for idea in ideas]
    similarity_matrix = cosine_similarity(embeddings)
    unique_ideas = []
    for i, idea in enumerate(ideas):
        if not any(similarity_matrix[i][j] > similarity_threshold for j in range(i) if i != j):
            unique_ideas.append(idea)
    print(f"Reduced to {len(unique_ideas)} unique ideas")
    return unique_ideas

def expand_idea(seed_idea, topic, context):
    """
    Expands a seed idea into a full research proposal.
    This function implements the Idea Expansion step from the paper,
    using chain-of-thought prompting to guide the expansion process.
    """
    print(f"Expanding seed idea: {seed_idea['title']}")
    prompt = f"""Expand the following seed idea into a full research proposal related to the topic "{topic}":

Seed Idea: {seed_idea['title']} - {seed_idea['description']}

Context:
{context}

Use the following steps to expand the idea:
1. Clearly define the problem and its significance
2. Analyze existing methods and their limitations
3. Explain the motivation behind your proposed approach
4. Describe your proposed method in detail, including potential challenges and how to address them

Provide the expanded idea in the following JSON format:
{{
    "title": "Title of the research idea",
    "problem_statement": "Brief statement of the problem being addressed",
    "existing_methods": "Overview of current approaches",
    "motivation": "Why this idea is important and novel",
    "proposed_method": "Detailed description of the proposed approach"
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert AI research assistant capable of expanding research ideas into full proposals."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

def rank_idea(idea):
    """
    Ranks an expanded idea based on novelty, feasibility, and potential impact.
    This function implements the Idea Ranking step from the paper,
    using chain-of-thought prompting to encourage detailed evaluation.
    """
    print(f"Ranking idea: {idea['title']}")
    prompt = f"""Evaluate the following research idea on a scale of 1-10 for each of novelty, feasibility, and potential impact. 
    Provide a brief justification for each score.

{json.dumps(idea, indent=2)}

Use the following steps for evaluation:
1. Assess novelty: Consider how different the idea is from existing work
2. Evaluate feasibility: Consider technical challenges, required resources, and timeline
3. Gauge potential impact: Consider both academic and practical implications
4. Provide an overall score as an average of the three dimensions

Respond in the following JSON format:
{{
    "novelty": {{
        "score": float,
        "justification": "string"
    }},
    "feasibility": {{
        "score": float,
        "justification": "string"
    }},
    "potential_impact": {{
        "score": float,
        "justification": "string"
    }},
    "overall_score": float
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert AI research evaluator capable of assessing the quality of research ideas."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    evaluation = json.loads(response.choices[0].message.content)
    return evaluation['overall_score'], json.dumps(evaluation, indent=2)

def human_rerank(ranked_ideas):
    """
    Simulates human reranking of the top ideas.
    In a real scenario, this function would involve actual human input.
    """
    print("Top 5 AI-ranked ideas:")
    for i, (idea, score, evaluation) in enumerate(ranked_ideas[:5], 1):
        print(f"{i}. Score: {score:.2f}")
        print(f"Title: {idea['title']}")
        print(f"Problem Statement: {idea['problem_statement'][:200]}...")
        print()
    
    print("Simulating human reranking...")
    # In a real scenario, we would implement a way for human input here
    # For simulation, we'll just shuffle the top 5
    top_5 = ranked_ideas[:5]
    np.random.shuffle(top_5)
    return top_5 + ranked_ideas[5:]

def generate_research_ideas(topic):
    """
    Main function that orchestrates the entire idea generation process.
    This function implements the full pipeline described in the paper:
    1. Retrieval of relevant papers
    2. Generation of seed ideas
    3. Deduplication of ideas
    4. Expansion of top ideas
    5. Ranking of expanded ideas
    6. Human reranking (simulated)
    """
    print(f"\n--- Generating research ideas for topic: {topic} ---")
    
    # Get relevant papers from Semantic Scholar
    papers = search_semantic_scholar(topic)
    
    if not papers:
        print(f"No papers found for topic: {topic}")
        return None
    
    print(f"Retrieved {len(papers)} relevant papers from Semantic Scholar")
    
    # Prepare context from the papers
    context = "\n".join([f"Title: {p['title']}\nAbstract: {p.get('abstract', 'N/A')}\nYear: {p.get('year', 'N/A')}\nCitations: {p.get('citationCount', 'N/A')}\n" for p in papers])

    # Generate seed ideas
    seed_ideas = generate_seed_ideas(topic, context, num_seeds=20)  # Reduced for demonstration, paper uses 4000
    
    # Deduplicate seed ideas
    unique_seeds = deduplicate_ideas([json.dumps(seed) for seed in seed_ideas])
    unique_seeds = [json.loads(seed) for seed in unique_seeds]
    
    # Expand ideas
    print("Expanding top 5 unique ideas...")
    expanded_ideas = [expand_idea(seed, topic, context) for seed in unique_seeds[:5]]  # Expand top 5 for demonstration
    
    # Rank ideas
    print("Ranking expanded ideas...")
    ranked_ideas = [(idea, *rank_idea(idea)) for idea in expanded_ideas]
    ranked_ideas.sort(key=lambda x: x[1], reverse=True)
    
    # Human reranking
    final_ranking = human_rerank(ranked_ideas)
    
    return final_ranking

def main():
    topics = [
        "Novel prompting methods to reduce hallucinations in large language models",
        "Improving factual consistency in language model outputs",
        "Enhancing cross-lingual transfer in low-resource languages",
        "Developing more efficient fine-tuning techniques for large language models",
        "Improving mathematical reasoning capabilities of language models",
        "Quantum computing applications in cryptography",
        "CRISPR gene editing for treating genetic disorders",
        "Nanomaterials for efficient solar energy conversion",
        "Machine learning in drug discovery and development",
        "Neuroplasticity and cognitive enhancement techniques",
        "Dark matter detection methods in astrophysics",
        "Microbiome manipulation for improving crop yields",
        "Sustainable alternatives to lithium-ion batteries",
        "Artificial photosynthesis for carbon capture",
        "Bioengineered organs for transplantation"
    ]
    
    all_generated_ideas = {}
    
    for topic in topics:
        ideas = generate_research_ideas(topic)
        if ideas:
            all_generated_ideas[topic] = ideas
            print(f"Generated {len(ideas)} ideas for topic: {topic}")
        else:
            print(f"Failed to generate ideas for topic: {topic}")
        sleep(1)  # To avoid hitting API rate limits
    
    # Save generated ideas to a file
    with open('generated_research_ideas.json', 'w') as f:
        json.dump(all_generated_ideas, f, indent=2)
    
    print("\nAll generated ideas have been saved to 'generated_research_ideas.json'")

if __name__ == "__main__":
    main()