import os
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI

# Set up OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Pydantic models for structured output
# These models represent the key components of the LoT approach:
# logical propositions, expressions, and the final output structure
class LogicalProposition(BaseModel):
    symbol: str
    description: str

class LogicalExpression(BaseModel):
    expression: str
    description: str

class LogicOfThoughtOutput(BaseModel):
    propositions: List[LogicalProposition] = Field(..., description="List of logical propositions")
    expressions: List[LogicalExpression] = Field(..., description="List of logical expressions")
    extended_expressions: List[LogicalExpression] = Field(..., description="List of extended logical expressions")
    translated_description: str = Field(..., description="Natural language description of the logical reasoning")
    solution: str = Field(..., description="Solution to the puzzle")

def logic_extraction(context: str) -> Dict[str, Any]:
    """
    Implements the Logic Extraction phase of LoT.
    This function extracts logical propositions and expressions from the input context.
    
    Why: This step is crucial for identifying the key logical components of the problem,
    which will be used for further reasoning.
    """
    prompt = f"""Extract logical propositions and expressions from the following context:

{context}

Provide the output as a JSON object with 'propositions' and 'expressions' keys.
Each proposition should be an object with 'symbol' and 'description' keys.
Each expression should be an object with 'expression' and 'description' keys.
Focus on extracting key relationships and constraints from the puzzle.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in logical reasoning and propositional logic."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

def logic_extension(expressions: List[LogicalExpression]) -> List[LogicalExpression]:
    """
    Implements the Logic Extension phase of LoT.
    This function expands the logical expressions using logical reasoning laws.
    
    Why: This step is essential for deriving new relationships and constraints
    that may not be explicitly stated in the original problem, but are logically implied.
    """
    prompt = f"""Extend the following logical expressions using logical reasoning laws:

{json.dumps([expr.model_dump() for expr in expressions])}

Provide the output as a JSON array of extended expressions with "extended_expressions" root key.
Each expression should be an object with 'expression' and 'description' keys.
Focus on deriving new relationships and constraints that can help solve the puzzle.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in logical reasoning and propositional logic."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    return [LogicalExpression(**expr) for expr in result.get('extended_expressions', [])]

def logic_translation(extended_expressions: List[LogicalExpression]) -> str:
    """
    Implements the Logic Translation phase of LoT.
    This function translates the extended logical expressions back into natural language.
    
    Why: This step is crucial for making the logical reasoning process understandable
    and interpretable, bridging the gap between formal logic and natural language understanding.
    """
    prompt = f"""Translate the following extended logical expressions into a natural language description:

{json.dumps([expr.model_dump() for expr in extended_expressions])}

Provide the output as a single string that explains the logical reasoning process.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert in logical reasoning and propositional logic."},
            {"role": "user", "content": prompt}
        ]
    )
    print(response)
    
    return response.choices[0].message.content.strip()

def solve_puzzle(context: str, translated_description: str) -> str:
    """
    Uses the original context and the translated logical reasoning to solve the puzzle.
    
    Why: This step combines the original problem statement with the derived logical insights
    to arrive at a solution, demonstrating the power of the LoT approach in solving complex
    logical reasoning tasks.
    """
    prompt = f"""Using the following context and logical reasoning:

Context:
{context}

Logical Reasoning:
{translated_description}

Solve the puzzle: Who will give a theatre ticket and who will buy roses?
Provide a step-by-step explanation of your reasoning, and then state the final answer.
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert puzzle solver using logical reasoning."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content.strip()

def run_logic_of_thought(context: str) -> LogicOfThoughtOutput:
    """
    Orchestrates the entire Logic-of-Thought process.
    
    This function ties together all the phases of LoT:
    1. Logic Extraction
    2. Logic Extension
    3. Logic Translation
    4. Puzzle Solving
    
    Why: This orchestration demonstrates how the LoT approach
    systematically builds up logical understanding to solve complex problems.
    """
    # Logic Extraction
    extraction_result = logic_extraction(context)
    propositions = [LogicalProposition(**prop) for prop in extraction_result.get('propositions', [])]
    expressions = [LogicalExpression(**expr) for expr in extraction_result.get('expressions', [])]
    
    # Logic Extension
    extended_expressions = logic_extension(expressions)
    
    # Logic Translation
    translated_description = logic_translation(extended_expressions)
    
    # Solve Puzzle
    solution = solve_puzzle(context, translated_description)
    
    return LogicOfThoughtOutput(
        propositions=propositions,
        expressions=expressions,
        extended_expressions=extended_expressions,
        translated_description=translated_description,
        solution=solution
    )

# Example puzzle (simplified version of Einstein's Riddle)
einsteins_riddle = """
The Simpsons are preparing a concert show with tricks, a guitar song, and one family member's own poem.
A person who wrote the poem will give a bread machine as a present and buy irises.
Mummy will buy tulips.
Melanie has learned to bake cinnamon buns and remembered guitar chords for Granny's birthday.
The trickster has prepared a notebook for recipes and a fruit salad.
Melanie knows that Granny likes daisies.
A person who will give a rocking chair will also prepare homemade candies for Granny.
Bill has a special deck of cards and a box with a double bottom.
Mummy and Melanie have had rehearsals for two for a week.
Daddy will prepare orange juice.
Granny will also be given a bouquet of roses and a ticket to the theatre play.

Who will give a theatre ticket and who will buy roses?
"""

def main():
    print("Solving Einstein's Riddle using Logic-of-Thought approach:")
    print("source credit: https://edcraft.io/blog/all-articles/5-zebra-puzzles-for-kids")
    print(einsteins_riddle)
    print("\nApplying Logic-of-Thought...")
    
    try:
        result = run_logic_of_thought(einsteins_riddle)
        
        print("\nLogic-of-Thought Output:")
        print(f"Propositions: {result.propositions}")
        print(f"Expressions: {result.expressions}")
        print(f"Extended Expressions: {result.extended_expressions}")
        print(f"\nLogical Reasoning Process:\n{result.translated_description}")
        print(f"\nSolution:\n{result.solution}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

"""
Core Concepts and Potential Improvements:

1. Logic-of-Thought (LoT) Approach:
   - Extracts logical propositions and expressions from natural language.
   - Extends logical expressions using reasoning laws.
   - Translates logical reasoning back into natural language.
   - Combines original context with derived logical insights to solve problems.

2. Use of Large Language Models (LLMs):
   - Leverages LLMs for each phase of the LoT process.
   - Demonstrates the power of LLMs in understanding and manipulating logical structures.

3. Structured Output:
   - Uses Pydantic models to ensure consistent and typed output structure.

Potential Improvements:
1. Implement more diverse logical connectives and reasoning laws.
2. Add error handling and validation for logical expressions.
3. Implement visualization of the logical reasoning process.
4. Optimize API calls to reduce token usage and improve efficiency.
5. Implement caching mechanisms for repeated logical structures.
6. Extend to handle more complex logical puzzles and real-world reasoning tasks.
7. Integrate with other AI techniques like knowledge graphs or automated theorem provers.
"""