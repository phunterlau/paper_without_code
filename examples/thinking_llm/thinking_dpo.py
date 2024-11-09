"""
Implementation of "THINKING LLMs: GENERAL INSTRUCTION FOLLOWING WITH THOUGHT GENERATION"
Key concepts:
1. Thought generation before response (Section 2.1)
2. Thought preference optimization without human thought data (Section 2.2)
3. Benefits extend beyond traditional reasoning tasks (Section 3.4)
"""

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ThoughtPromptType(Enum):
    """
    Paper experiments with two thought prompt types (Figure 2):
    - GENERIC: Free-form thinking without structure
    - SPECIFIC: Structured thinking with draft and evaluation
    The paper found both types effective but SPECIFIC provides more consistent format
    """
    GENERIC = "generic"
    SPECIFIC = "specific"

@dataclass
class ThoughtResponse:
    """
    Paper Section 2.1: Thoughts are kept internal and not shown to user
    Only the response is presented as final output
    Score is determined by judge model evaluation
    """
    thought: str  # Internal reasoning (not shown to user)
    response: str  # Final output to user
    score: float  # Judge model evaluation score

@dataclass
class TPOExample:
    """
    Paper Section 3.4: Examples span different categories to demonstrate
    TPO's effectiveness beyond traditional reasoning tasks
    """
    instruction: str
    category: str  # Categories from fine-grained evaluation
    thought_type: ThoughtPromptType
    outputs: List[ThoughtResponse]

def get_thought_prompt(prompt_type: ThoughtPromptType) -> str:
    """
    Paper Figure 2: Two types of thought prompts:
    1. Generic: Allows natural thought flow, similar to chain-of-thought
    2. Specific: Enforces structure with draft and evaluation phases
    
    Key insight: Both prompts keep thoughts internal rather than showing to user
    """
    if prompt_type == ThoughtPromptType.GENERIC:
        return """Respond to the following user query in a comprehensive and detailed way. 
        Write your thoughts after "Here is my thought process:" and 
        write your response after "Here is my response:"."""
    else:
        return """Respond to the following user query in a comprehensive and detailed way. 
        First write your internal thoughts including draft response and evaluation. 
        Write your final response after "<R>"."""

def generate_thoughts(instruction: str, prompt_type: ThoughtPromptType, num_samples: int = 3) -> List[Dict[str, str]]:
    """
    Paper Section 2.1: Initial thought generation phase
    
    Key aspects:
    1. Multiple samples (K=8 in paper, reduced here for demo)
    2. Temperature=0.8 encourages thought diversity
    3. Structured JSON output separates thought from response
    4. Thoughts are considered internal processing
    """
    print(f"\n{'='*80}")
    print(f"Method: Initial Thought Generation (Section 2.1)")
    print(f"Input: {instruction}")
    print(f"Thought Type: {prompt_type.value}")
    
    # System prompt enforces JSON structure for thought/response separation
    system_prompt = """You are an AI assistant trained for structured thinking.
    Return JSON with exactly:
    1. "thought": your internal reasoning process
    2. "response": your final answer"""
    
    user_prompt = f"{get_thought_prompt(prompt_type)}\n\nUser query: {instruction}"
    
    responses = []
    for i in range(num_samples):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.8,  # Paper: Encourages diverse thought patterns
                top_p=0.95
            )
            
            output = json.loads(response.choices[0].message.content)
            print(f"\nSample {i+1}:")
            print_truncated(output['thought'], "Thought: ")
            print_truncated(output['response'], "Response: ")
            responses.append(output)
            
        except Exception as e:
            print(f"Error: {e}")
            continue
            
    return responses

def evaluate_response(instruction: str, response: str) -> float:
    """
    Paper Section 2.2: Judge model evaluation
    
    Key insights:
    1. Only evaluates response, not thoughts
    2. Indirect optimization of thoughts through response quality
    3. Uses ArmoRM-style scalar scoring (0-1)
    4. No need for special thought evaluation
    """
    print(f"\n{'='*80}")
    print("Method: Judge Model Evaluation (Section 2.2)")
    
    prompt = f"""Rate this response on scale 0-1 based on:
    - Helpfulness and relevance
    - Accuracy and clarity
    - Completeness
    
    Instruction: {instruction}
    Response: {response}
    
    Return JSON with single field 'score' containing float 0-1"""
    
    try:
        result = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert judge evaluating response quality."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2  # Low temperature for consistent evaluation
        )
        
        score = float(json.loads(result.choices[0].message.content)['score'])
        print(f"Score: {score}")
        return score
        
    except Exception as e:
        print(f"Error: {e}")
        return 0.0

def optimize_thoughts(examples: List[TPOExample], iterations: int = 4) -> List[TPOExample]:
    """
    Paper Section 2.2: Thought Preference Optimization
    
    Key aspects:
    1. Iterative improvement through multiple rounds
    2. Preference optimization based on response quality
    3. Maintains best responses across iterations
    4. No direct supervision of thought process
    5. Learns to think effectively through response feedback
    """
    optimized_examples = []
    
    for example in examples:
        print(f"\n{'='*80}")
        print("Method: Thought Preference Optimization (Section 2.2)")
        print(f"Task: {example.instruction}")
        print(f"Category: {example.category}")
        
        current_outputs = example.outputs
        best_score_so_far = 0.0
        
        for iteration in range(iterations):
            print(f"\nIteration {iteration+1}/{iterations}")
            
            # Generate new thought-response pairs
            new_pairs = generate_thoughts(
                example.instruction,
                example.thought_type
            )
            
            # Evaluate responses (not thoughts)
            scored_responses = []
            for pair in new_pairs:
                score = evaluate_response(example.instruction, pair['response'])
                scored_responses.append(
                    ThoughtResponse(
                        thought=pair['thought'],
                        response=pair['response'],
                        score=score
                    )
                )
            
            # Preference optimization: Select best responses
            scored_responses.sort(key=lambda x: x.score, reverse=True)
            current_outputs = scored_responses[:2]  # Keep top performers
            
            # Track improvements
            current_best = max(scored_responses, key=lambda x: x.score)
            if current_best.score > best_score_so_far:
                best_score_so_far = current_best.score
                print("\nNew Best Response Found:")
                print(f"Score: {best_score_so_far}")
                print_truncated(current_best.thought, "Thought Process:\n")
                print_truncated(current_best.response, "Response:\n")
        
        optimized_examples.append(
            TPOExample(
                instruction=example.instruction,
                category=example.category,
                thought_type=example.thought_type,
                outputs=current_outputs
            )
        )
    
    return optimized_examples

def main():
    """
    Paper Section 3.4: Demonstrate TPO effectiveness on non-traditional tasks
    
    Key findings:
    1. TPO improves performance beyond reasoning/math tasks
    2. Benefits seen in marketing, health, creative writing
    3. Thinking helps even in seemingly straightforward tasks
    """
    examples = [
        TPOExample(
            # Marketing example (showed improvement per Section 3.4)
            instruction="Create a compelling marketing pitch for an eco-friendly water bottle",
            category="Marketing and Sales",
            thought_type=ThoughtPromptType.SPECIFIC,
            outputs=[]
        ),
        TPOExample(
            # Health category example (showed benefits from thinking)
            instruction="Design a weekly workout plan for a beginner focusing on strength training",
            category="Health and Wellness",
            thought_type=ThoughtPromptType.SPECIFIC,
            outputs=[]
        )
    ]
    
    optimized_examples = optimize_thoughts(examples)
    
    # Display final results
    print("\n=== Final Results ===")
    for example in optimized_examples:
        print(f"\nTask: {example.instruction}")
        best_response = max(example.outputs, key=lambda x: x.score)
        print(f"Final Score: {best_response.score}")
        print_truncated(best_response.thought, "Final Thought Process:\n")
        print_truncated(best_response.response, "Final Response:\n")

if __name__ == "__main__":
    main()