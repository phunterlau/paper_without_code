import os
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI

# Set up OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class Date(BaseModel):
    month: str
    day: int

class LogicOfThoughtOutput(BaseModel):
    possible_dates: List[Date]
    propositions: List[str]
    logical_reasoning: List[str]
    solution: Date

def logic_extraction(dates: List[Date]) -> List[str]:
    """
    Extracts logical propositions from the given dates.
    """
    date_strings = [f"{date.month} {date.day}" for date in dates]
    prompt = f"""
    Given the following possible dates for Cheryl's birthday:
    {', '.join(date_strings)}

    Extract logical propositions about these dates, considering:
    1. The uniqueness of days and months
    2. Albert knows the month, Bernard knows the day
    3. Albert's first statement: "I don't know when Cheryl's birthday is, but I know that Bernard doesn't know too."
    4. Bernard's statement: "At first I didn't know when Cheryl's birthday is, but I know now."
    5. Albert's second statement: "Then I also know when Cheryl's birthday is."

    Provide the output as a JSON object with a 'propositions' key containing an array of strings, each string being a logical proposition.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in logical reasoning and propositional logic."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    return result.get('propositions', [])

def logic_extension(propositions: List[str]) -> List[str]:
    prompt = f"""
    Given the following logical propositions:
    {json.dumps(propositions)}

    Extend these propositions using logical reasoning laws to deduce Cheryl's birthday.
    Follow these steps carefully:
    1. Do not eliminate any months based solely on Albert's first statement.
    2. Identify all unique days across all months.
    3. Based on Albert's first statement, focus on months that do not have unique days.
    4. Based on Bernard's statement, focus on days that are unique among the remaining months.
    5. Apply Albert's final statement to determine which of these days also allows Albert to know the month.
    6. Ensure that the final deduction leaves only one possible date.
    7. Double-check that your final answer is consistent with all given statements and is actually in the list of possible dates.

    Provide a step-by-step logical reasoning process as a JSON object with a 'reasoning_steps' key containing an array of strings.
    Make sure the final step clearly states the correct birthday based on all given information.
    Double-check that your reasoning doesn't make any unfounded assumptions about eliminating months or days.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in logical reasoning and propositional logic. Ensure your reasoning is precise and follows from the given statements without introducing external assumptions."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    return result.get('reasoning_steps', [])

def solve_puzzle(dates: List[Date], logical_reasoning: List[str]) -> Date:
    prompt = f"""
    Given the following logical reasoning process:
    {json.dumps(logical_reasoning)}

    And the following possible dates:
    {', '.join([f"{date.month} {date.day}" for date in dates])}

    Determine Cheryl's birthday by following these steps:
    1. Identify months that do not have unique days.
    2. Among these months, find days that are unique.
    3. Determine which of these unique days allows both Bernard to know the date and Albert to know the month.
    4. Verify that this date is consistent with all given statements.
    5. Ensure that your solution is one of the possible dates provided.

    Provide the answer as a JSON object with 'solution' key containing 'month' and 'day' subkeys.
    If no unique solution can be found or the solution is not in the list of possible dates, return null.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert puzzle solver using logical reasoning."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    solution = result.get('solution')
    if solution and solution.get('month') and solution.get('day'):
        proposed_solution = Date(month=solution['month'], day=int(solution['day']))
        if proposed_solution in dates:
            return proposed_solution
    return None

def run_logic_of_thought(dates: List[Date]) -> LogicOfThoughtOutput:
    # Logic Extraction
    propositions = logic_extraction(dates)
    print("Extracted Propositions:")
    print(json.dumps(propositions, indent=2))
    
    # Logic Extension
    logical_reasoning = logic_extension(propositions)
    print("\nLogical Reasoning:")
    print(json.dumps(logical_reasoning, indent=2))
    
    # Solve Puzzle
    solution = solve_puzzle(dates, logical_reasoning)
    
    if solution is None:
        print("Warning: No valid solution found.")
        return None
    
    # Verify the solution
    if solution not in dates:
        print(f"Warning: Solution {solution.month} {solution.day} is not in the list of possible dates.")
        return None
    
    print(f"\nSolution found: {solution.month} {solution.day}")
    
    return LogicOfThoughtOutput(
        possible_dates=dates,
        propositions=propositions,
        logical_reasoning=logical_reasoning,
        solution=solution
    )

def main():
    # Original dates
    original_dates = [
        Date(month="May", day=15), Date(month="May", day=16), Date(month="May", day=19),
        Date(month="June", day=17), Date(month="June", day=18),
        Date(month="July", day=14), Date(month="July", day=16),
        Date(month="August", day=14), Date(month="August", day=15), Date(month="August", day=17)
    ]

    # Gabe's dates
    gabe_dates = [
        Date(month="January", day=15), Date(month="January", day=4),
        Date(month="July", day=13), Date(month="July", day=24), Date(month="July", day=30),
        Date(month="March", day=13), Date(month="March", day=24),
        Date(month="May", day=11), Date(month="May", day=17), Date(month="May", day=30)
    ]

    # Another set of dates
    another_dates = [
        Date(month="April", day=23), Date(month="April", day=24), Date(month="April", day=29),
        Date(month="August", day=19), Date(month="August", day=24), Date(month="August", day=29),
        Date(month="June", day=13), Date(month="June", day=23),
        Date(month="March", day=14), Date(month="March", day=15),
        Date(month="May", day=13), Date(month="May", day=27)
    ]

    example_index = 0
    for dates in [original_dates, gabe_dates, another_dates]:
        example_index += 1
        # Choose which date set to use
        #dates = original_dates  # Change this to use different date sets

        print("\n" + "="*50)
        # get the index of the current dates
        print(f"Example No. {example_index}")

        print("Solving Cheryl's Birthday puzzle using Logic-of-Thought approach:")
        print("Possible dates:", ", ".join([f"{date.month} {date.day}" for date in dates]))
        
        try:
            result = run_logic_of_thought(dates)
            
            if result:
                print("\nFinal Logic-of-Thought Output:")
                print(json.dumps({
                    "propositions": result.propositions,
                    "logical_reasoning": result.logical_reasoning,
                    "solution": f"{result.solution.month} {result.solution.day}"
                }, indent=2))
            else:
                print("\nNo valid solution found.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()