import os
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI

# Set up OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Pydantic models for structured output
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
    solution: str = Field(..., description="Solution to Cheryl's Birthday puzzle")

def logic_extraction(context: str) -> Dict[str, Any]:
    """
    Implements the Logic Extraction phase of LoT.
    """
    prompt = f"""Extract logical propositions and expressions from the following context:

{context}

IMPORTANT: Only use dates that are explicitly listed in the context. 
Do not introduce any dates that are not in the given list.

Format your response as a JSON object with 'propositions' and 'expressions' keys.
Each proposition should be an object with 'symbol' and 'description' keys.
Each expression should be an object with 'expression' and 'description' keys.
Focus on extracting key relationships and constraints from the puzzle."""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in logical reasoning and propositional logic. Provide your response in JSON format."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

def logic_extension(expressions: List[LogicalExpression]) -> List[LogicalExpression]:
    """
    Implements the Logic Extension phase of LoT.
    This function expands the logical expressions using logical reasoning laws.
    """
    prompt = f"""Extend the following logical expressions using logical reasoning laws:

{json.dumps([expr.model_dump() for expr in expressions])}

Return a JSON object with an "extended_expressions" array.
Each expression in the array should be an object with 'expression' and 'description' keys.
Focus on deriving new relationships and constraints that can help solve the puzzle."""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in logical reasoning and propositional logic. Respond with a JSON object."},
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
    """
    prompt = f"""Translate the following extended logical expressions into a natural language description:

{json.dumps([expr.model_dump() for expr in extended_expressions])}

Provide your response as a JSON object with a 'reasoning' key containing a string with the natural language explanation."""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in logical reasoning and propositional logic. Return your explanation in JSON format."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    # Ensure we get a string from the 'reasoning' key
    if isinstance(result.get('reasoning'), (dict, list)):
        return str(result['reasoning'])
    return result.get('reasoning', '')

def solve_puzzle(context: str, translated_description: str, possible_dates: list) -> str:
    """
    Uses the original context and the translated logical reasoning to solve the puzzle.
    """
    prompt = f"""Using the following context and logical reasoning:

Context:
{context}

Logical Reasoning:
{translated_description}

The solution MUST be one of these exact dates: {', '.join(possible_dates)}

Return a JSON object with a 'solution' key containing your final answer and explanation for Cheryl's birthday."""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert puzzle solver using logical reasoning. Provide your solution in JSON format."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    # Ensure we get a string from the 'solution' key
    if isinstance(result.get('solution'), (dict, list)):
        return str(result['solution'])
    return result.get('solution', '')

def run_logic_of_thought(context: str, possible_dates: list) -> LogicOfThoughtOutput:
    """
    Orchestrates the entire Logic-of-Thought process.
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
    solution = solve_puzzle(context, translated_description, possible_dates)
    
    return LogicOfThoughtOutput(
        propositions=propositions,
        expressions=expressions,
        extended_expressions=extended_expressions,
        translated_description=translated_description,
        solution=solution
    )

def format_dates_by_month(dates):
    """
    Organizes dates by month for better readability in the puzzle.
    Args:
        dates: List or set of dates in format 'Month Day'
    Returns:
        String with dates organized by month
    """
    dates = list(dates)
    month_groups = {}
    for date in dates:
        month, day = date.split()
        if month not in month_groups:
            month_groups[month] = []
        month_groups[month].append(int(day))
    
    sorted_months = sorted(month_groups.keys(), 
                         key=lambda x: ['January', 'February', 'March', 'April', 'May', 
                                      'June', 'July', 'August', 'September', 'October', 
                                      'November', 'December'].index(x))
    
    formatted_lines = []
    for month in sorted_months:
        sorted_days = sorted(month_groups[month])
        days_str = ', '.join(str(day) for day in sorted_days)
        formatted_lines.append(f"{month} {days_str}")
    
    return '\n'.join(formatted_lines)

def get_puzzle_text(dates):
    """
    Generates the puzzle text with the given dates.
    Args:
        dates: List or set of dates in format 'Month Day'
    Returns:
        Complete puzzle text
    """
    formatted_dates = format_dates_by_month(dates)
    return f"""
Albert and Bernard just became friends with Cheryl, and they want to know when her birthday is.
Cheryl gave them a list of {len(dates)} possible dates:
{formatted_dates}

Cheryl then tells Albert only the month, and Bernard only the day, of her birthday.
Albert: I don't know when Cheryl's birthday is, but I know that Bernard doesn't know too.
Bernard: At first I didn't know when Cheryl's birthday is, but I know now.
Albert: Then I also know when Cheryl's birthday is.

What is Cheryl's birthday?
"""

def main():
    # Define date combinations
    date_combinations = {
        "Original": [
            'May 15', 'May 16', 'May 19',
            'June 17', 'June 18',
            'July 14', 'July 16',
            'August 14', 'August 15', 'August 17'
        ], # July 16
        "Gabe's": [
            'January 15', 'January 4',
            'July 13',    'July 24',   'July 30',
            'March 13',   'March 24',
            'May 11',     'May 17',    'May 30'
        ], #July 30
        "Alternative": {
            'April 23', 'April 24', 'April 29',
            'August 19', 'August 24', 'August 29',
            'June 13', 'June 23',
            'March 14', 'March 15',
            'May 13', 'May 27'
        } # June 13
    }
    
    # Process each combination
    for name, dates in date_combinations.items():
        print(f"\n{'='*80}")
        print(f"Processing {name} Date Combination")
        print('='*80)
        
        # Generate puzzle with current dates
        puzzle = get_puzzle_text(dates)
        
        print("Puzzle Configuration:")
        print(puzzle)
        print("\nApplying Logic-of-Thought...")
        
        try:
            result = run_logic_of_thought(puzzle, dates)
            
            print("\nLogic-of-Thought Output:")
            print("\nPropositions:")
            for prop in result.propositions:
                print(f"{prop.symbol}: {prop.description}")
            
            print("\nLogical Expressions:")
            for expr in result.expressions:
                print(f"{expr.expression}: {expr.description}")
            
            print("\nExtended Expressions:")
            for expr in result.extended_expressions:
                print(f"{expr.expression}: {expr.description}")
            
            print(f"\nReasoning Process:\n{result.translated_description}")
            print(f"\nSolution:\n{result.solution}")
            
        except Exception as e:
            print(f"An error occurred while processing {name} combination: {str(e)}")
            continue
        
        print(f"\nCompleted processing {name} combination")

if __name__ == "__main__":
    main()