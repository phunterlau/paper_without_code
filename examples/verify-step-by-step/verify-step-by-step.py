import os
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

# Set up OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Define Pydantic models for structured output
# These models ensure type safety and provide a clear structure for our data
class Step(BaseModel):
    content: str
    is_correct: Optional[bool] = None  # Allows for uncertainty in correctness
    feedback: str  # Stores feedback for each step
    improvement_suggestion: Optional[str] = None  # Allows for specific improvement suggestions
    uncertainty: float = Field(0.0, ge=0.0, le=1.0)  # Quantifies uncertainty in the step

class Solution(BaseModel):
    steps: List[Step]
    final_answer: str
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)  # Overall confidence in the solution

class ProblemContext(BaseModel):
    problem: str
    domain: str
    difficulty: int = Field(1, ge=1, le=10)  # Allows for problem difficulty scaling
    previous_attempts: List[Solution] = []  # Stores previous solution attempts for iterative improvement

def sanitize_step(step: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitizes the step data to ensure all required fields are present.
    This is crucial for maintaining data integrity throughout the process.
    """
    return {
        "content": step.get("content", ""),
        "is_correct": step.get("is_correct"),
        "feedback": step.get("feedback", ""),
        "improvement_suggestion": step.get("improvement_suggestion"),
        "uncertainty": step.get("uncertainty", 0.5)
    }

def process_supervision_batch(solution: Solution) -> Solution:
    """
    Implements process supervision by evaluating each step of the solution.
    This function is key to improving chain of thought reasoning by providing
    detailed feedback and improvement suggestions for each step.
    """
    try:
        steps_content = [step.content for step in solution.steps]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Evaluate each step of the solution. Provide detailed feedback, including whether the step is correct, any errors, and specific suggestions for improvement. Even for correct steps, suggest ways to make them clearer or more efficient."},
                {"role": "user", "content": f"Steps to evaluate: {json.dumps(steps_content)}"}
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "evaluate_steps",
                    "description": "Evaluate if each step is correct, provide detailed feedback, and suggest specific improvements",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "evaluations": {
                                "type": "array",
                                "items": Step.model_json_schema()
                            }
                        }
                    }
                }
            }],
            tool_choice={"type": "function", "function": {"name": "evaluate_steps"}}
        )
        
        evaluations = json.loads(response.choices[0].message.tool_calls[0].function.arguments)["evaluations"]
        supervised_steps = [Step(**eval) for eval in evaluations]
        
        return Solution(steps=supervised_steps, final_answer=solution.final_answer, confidence_score=solution.confidence_score)
    except Exception as e:
        print(f"Error in process supervision: {e}")
        return solution

def generate_solution(context: ProblemContext, previous_solution: Optional[Solution] = None) -> Solution:
    """
    Generates a solution based on the problem context and previous attempts.
    This function implements the iterative improvement aspect of the process,
    using feedback from previous attempts to guide the generation of new solutions.
    """
    try:
        previous_feedback = ""
        if previous_solution:
            previous_feedback = "Previous attempt feedback and improvements:\n" + "\n".join([
                f"Step {i+1}: {step.feedback}\nImprovement suggestion: {step.improvement_suggestion}"
                for i, step in enumerate(previous_solution.steps)
            ])

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are an AI assistant skilled in solving {context.domain} problems. Consider the feedback and improvement suggestions from the previous attempt to improve your solution."},
                {"role": "user", "content": f"Solve this problem step-by-step: {context.problem}\n\n{previous_feedback}"}
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "provide_solution",
                    "description": "Provide a step-by-step solution to the problem, implementing the suggested improvements from the previous attempt",
                    "parameters": Solution.model_json_schema()
                }
            }],
            tool_choice={"type": "function", "function": {"name": "provide_solution"}}
        )
        
        raw_solution = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        sanitized_steps = [Step(**sanitize_step(step)) for step in raw_solution.get("steps", [])]
        final_answer = raw_solution.get("final_answer", "No final answer provided.")
        
        return Solution(steps=sanitized_steps, final_answer=final_answer, confidence_score=0.0)
    except Exception as e:
        print(f"Error in generating solution: {e}")
        return Solution(steps=[], final_answer="Error in solution generation", confidence_score=0.0)

def solve_problem_with_process_supervision(context: ProblemContext, max_iterations: int = 3) -> Solution:
    """
    Main function that implements the iterative problem-solving process with process supervision.
    This function embodies the core idea of improving chain of thought reasoning through
    repeated attempts and detailed feedback.
    """
    best_solution = None
    best_score = float('-inf')

    for i in range(max_iterations):
        print(f"\nIteration {i+1}")
        
        # Generate a new solution, potentially improving upon the previous best
        solution = generate_solution(context, best_solution)
        
        # Apply process supervision to evaluate and improve the solution
        supervised_solution = process_supervision_batch(solution)
        supervised_solution.confidence_score = calculate_confidence_score(supervised_solution)
        
        print(f"\nSolution for iteration {i+1}:")
        print(json.dumps(supervised_solution.model_dump(), indent=2))
        
        # Update the best solution if the current one is better
        if supervised_solution.confidence_score > best_score:
            best_solution = supervised_solution
            best_score = supervised_solution.confidence_score
        
        # Check if we've reached a correct solution
        if all(step.is_correct for step in supervised_solution.steps):
            print("Solution is correct according to process supervision!")
            return supervised_solution
        
        if i < max_iterations - 1:
            print("Preparing to improve solution based on process supervision feedback...")
        else:
            print(f"Reached maximum iterations ({max_iterations}). Returning best solution.")
    
    return best_solution

def calculate_confidence_score(solution: Solution) -> float:
    """
    Calculates a confidence score for the solution based on the correctness of its steps.
    This provides a quantitative measure of solution quality.
    """
    correct_steps = sum(1 for step in solution.steps if step.is_correct)
    total_steps = len(solution.steps)
    return correct_steps / total_steps if total_steps > 0 else 0.0


if __name__ == "__main__":
    # Define a set of diverse problems to test the system's capabilities
    problems = [
        ProblemContext(problem="Solve the equation: 3x - 7 = 20", domain="algebra", difficulty=3),
        ProblemContext(problem="Find the derivative of f(x) = x^3 - 2x + 1", domain="calculus", difficulty=4),
        ProblemContext(problem="Calculate the probability of drawing two aces from a standard deck of 52 cards without replacement", domain="probability", difficulty=5),
        ProblemContext(problem="Explain the process of photosynthesis in plants, breaking it down into key steps.", domain="biology", difficulty=6),
        ProblemContext(problem="Analyze the poem 'The Road Not Taken' by Robert Frost, explaining its literary devices and themes.", domain="literature", difficulty=7)
    ]
    
    # Solve each problem using the process supervision approach
    for i, problem_context in enumerate(problems, 1):
        print(f"\n{'='*50}\nProblem {i}: {problem_context.problem}")
        final_solution = solve_problem_with_process_supervision(problem_context, max_iterations=3)
        print("\nFinal solution:")
        print(json.dumps(final_solution.model_dump(), indent=2))

"""
Key Steps and Future Improvements:

1. Problem Representation: The script uses structured models (ProblemContext, Solution, Step) to represent problems and solutions. This allows for clear organization and easy manipulation of data.

2. Process Supervision: The process_supervision_batch function implements detailed evaluation of each step, providing feedback and improvement suggestions. This is crucial for improving chain of thought reasoning.

3. Iterative Improvement: The solve_problem_with_process_supervision function implements an iterative approach, generating new solutions based on feedback from previous attempts.

4. Confidence Scoring: The calculate_confidence_score function provides a quantitative measure of solution quality, allowing for comparison between iterations.

Future Improvements:
1. Dynamic Iteration Control: Implement a more sophisticated method to determine when to stop iterating, possibly based on diminishing returns in improvement.

2. Meta-learning: Incorporate a mechanism to learn from solving multiple problems, improving the initial solution generation over time.

3. Explanation Generation: Add a feature to generate human-readable explanations of how and why the solution improved over iterations.

4. Uncertainty Handling: Make better use of the uncertainty field in the Step model, possibly to guide the focus of improvement efforts.

5. Domain-Specific Strategies: Implement specialized strategies for different problem domains to improve solution quality and efficiency.

6. Interactive Mode: Develop an interactive mode where a human can provide additional guidance or corrections during the solution process.

7. Parallel Processing: Implement parallel processing of multiple solution attempts to increase efficiency and explore a wider solution space.
"""