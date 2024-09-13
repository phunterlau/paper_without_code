"""
Q* Algorithm Implementation

Overview:
This implementation demonstrates the Q* algorithm, a novel approach to enhance multi-step reasoning
in Large Language Models (LLMs). The core idea is to formulate the reasoning process as a Markov
Decision Process (MDP) and use a heuristic search algorithm (inspired by A*) to guide the LLM
through the problem-solving steps.

Key components:
1. MathProblemSolver: Simulates an LLM solving math problems step-by-step.
2. QValueEstimator: Estimates the value of each potential next step.
3. QStar: Implements the main Q* algorithm, using beam search and Q-value guidance.

The algorithm works by generating multiple possible solution paths, evaluating them using
estimated Q-values, and selecting the most promising path at each step. This approach allows
for more deliberate and accurate problem-solving compared to standard auto-regressive LLM generation.

Thought flow:
1. Initialize the problem state
2. Generate possible next steps
3. Estimate Q-values for each step
4. Select the best steps using beam search
5. Repeat steps 2-4 until a solution is found or max steps are reached
6. Verify and potentially correct the final solution

This implementation serves as a proof of concept and can be extended to various reasoning tasks
beyond math problem solving.
"""

import os
import json
from typing import List, Optional
from pydantic import BaseModel
from openai import OpenAI
import re

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class MathStep(BaseModel):
    step: str
    explanation: str

class MathSolution(BaseModel):
    steps: List[MathStep]
    final_answer: str
    numerical_answer: Optional[float] = None

class QValue(BaseModel):
    state: str
    action: str
    value: float

class MathProblemSolver:
    def solve(self, question: str) -> MathSolution:
        """
        Simulates an LLM solving a math problem step-by-step.
        
        Why here: This method encapsulates the LLM's problem-solving capability,
        allowing us to use it as a black box within the Q* algorithm. It's crucial
        for generating initial solutions and potential next steps.
        """
        print(f"MathProblemSolver: Solving question - {question}")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert math problem solver. Provide a step-by-step solution focusing on direct, logical steps leading to the final answer in the requested units."},
                {"role": "user", "content": f"Solve this math problem step by step and provide a clear numerical answer in the requested units: {question}"}
            ],
            functions=[{
                "name": "provide_solution",
                "description": "Provide a step-by-step solution to a math problem",
                "parameters": MathSolution.model_json_schema()
            }],
            function_call={"name": "provide_solution"}
        )
        solution = MathSolution.model_validate_json(response.choices[0].message.function_call.arguments)
        print(f"MathProblemSolver: Solution found with {len(solution.steps)} steps")
        return solution

class QValueEstimator:
    def estimate(self, state: str, action: str) -> QValue:
        """
        Estimates the Q-value for a given state-action pair.
        
        Why here: Q-value estimation is central to the Q* algorithm. It provides
        a heuristic for evaluating the potential of each step, guiding the search
        towards more promising solutions. This method allows us to leverage the
        LLM's knowledge for this crucial evaluation.
        """
        print(f"QValueEstimator: Estimating Q-value for action - {action[:30]}...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in evaluating problem-solving steps. Focus on steps that directly lead to the solution in the requested units."},
                {"role": "user", "content": f"Estimate the likelihood (0-1) that this step will lead to the correct solution in the requested units.\nCurrent state: {state}\nProposed step: {action}"}
            ],
            functions=[{
                "name": "estimate_q_value",
                "description": "Estimate the Q-value for a state-action pair",
                "parameters": QValue.model_json_schema()
            }],
            function_call={"name": "estimate_q_value"}
        )
        q_value = QValue.model_validate_json(response.choices[0].message.function_call.arguments)
        print(f"QValueEstimator: Q-value estimated as {q_value.value}")
        return q_value

class QStar:
    def __init__(self, solver: MathProblemSolver, q_estimator: QValueEstimator):
        self.solver = solver
        self.q_estimator = q_estimator

    def solve(self, question: str, max_steps: int = 7, beam_width: int = 3) -> MathSolution:
        """
        Implements the Q* algorithm to solve a math problem.
        
        Why here: This is the core of the Q* algorithm. It orchestrates the entire
        problem-solving process, combining beam search with Q-value guidance to
        explore and evaluate multiple solution paths simultaneously.
        """
        print(f"QStar: Starting to solve - {question}")
        initial_state = question
        beam = [(initial_state, [], 0)]  # Each element is (state, path, score)
        seen_steps = set()
        best_solution = None

        for step in range(max_steps):
            print(f"QStar: Step {step + 1}/{max_steps}")
            candidates = []
            for state, path, score in beam:
                # Generate and evaluate possible next steps
                next_steps = self._generate_next_steps(state, seen_steps)
                for next_step in next_steps:
                    if next_step.step not in seen_steps:
                        new_state = f"{state}\n{next_step.step}"
                        q_value = self.q_estimator.estimate(state, next_step.step)
                        new_score = score + q_value.value
                        candidates.append((new_state, path + [next_step], new_score))
                        seen_steps.add(next_step.step)
            
            if not candidates:
                print("QStar: No more candidates, breaking")
                break

            # Select top candidates (beam search)
            beam = sorted(candidates, key=lambda x: x[2], reverse=True)[:beam_width]
            print(f"QStar: Top candidate score: {beam[0][2]}")

            # Check if a complete solution is found
            current_solution = self._format_solution(beam[0][1])
            if current_solution.numerical_answer is not None:
                best_solution = current_solution
                if self._is_complete_solution(question, beam[0][0], current_solution):
                    print("QStar: Complete solution found")
                    return best_solution

        # If no complete solution found, use the best partial solution
        if best_solution is None:
            print("QStar: No complete solution found, using best partial solution")
            best_solution = self._format_solution(beam[0][1])
        
        # Verify and potentially correct the solution
        print("QStar: Verifying and potentially correcting the solution")
        verified_solution = self._verify_and_correct_solution(question, best_solution)
        return verified_solution

    def _generate_next_steps(self, state: str, seen_steps: set) -> List[MathStep]:
        """
        Generates possible next steps for the problem-solving process.
        
        Why here: This method is crucial for exploring the solution space. By leveraging
        the LLM to generate diverse and relevant steps, we can consider multiple
        problem-solving approaches, enhancing the algorithm's ability to find optimal solutions.
        """
        print("QStar: Generating next steps")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert math problem solver. Generate possible next steps, focusing on reaching a numerical answer in the requested units. Avoid repeating steps or adding unnecessary ones."},
                {"role": "user", "content": f"Generate 3 possible next steps for this math problem, focusing on reaching a numerical answer in the requested units:\n{state}\n\nAvoid these steps: {seen_steps}"}
            ],
            functions=[{
                "name": "generate_steps",
                "description": "Generate possible next steps for a math problem",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": MathStep.model_json_schema()
                        }
                    },
                    "required": ["steps"]
                }
            }],
            function_call={"name": "generate_steps"}
        )
        
        function_call = response.choices[0].message.function_call
        if function_call and function_call.arguments:
            try:
                args = json.loads(function_call.arguments)
                steps = [MathStep.model_validate(step) for step in args.get("steps", [])]
            except json.JSONDecodeError:
                print("QStar: Error decoding JSON in _generate_next_steps")
                steps = []
        else:
            print("QStar: No function call in _generate_next_steps response")
            steps = []

        print(f"QStar: Generated {len(steps)} next steps")
        for i, step in enumerate(steps, 1):
            print(f"  Step {i}: {step.step}")
            print(f"    Explanation: {step.explanation}")
        return steps

    def _is_complete_solution(self, question: str, state: str, solution: MathSolution) -> bool:
        """
        Checks if the current solution is complete and correct.
        
        Why here: This method is essential for determining when to stop the search process.
        It ensures that we not only have a numerical answer but also that the solution
        directly addresses the original question, preventing unnecessary additional steps.
        """
        print("QStar: Checking if solution is complete")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in evaluating math solutions. Ensure the solution directly answers the question in the requested units."},
                {"role": "user", "content": f"Does this represent a complete solution to the math problem with a clear numerical answer in the requested units? Answer with 'Yes' or 'No' and explain briefly.\n\nQuestion: {question}\n\nSolution:\n{state}\nFinal Answer: {solution.final_answer}\nNumerical Answer: {solution.numerical_answer}"}
            ]
        )
        answer = response.choices[0].message.content.strip().lower()
        is_complete = "yes" in answer and solution.numerical_answer is not None
        print(f"QStar: Solution completeness: {is_complete}")
        return is_complete

    def _format_solution(self, steps: List[MathStep]) -> MathSolution:
        """
        Formats the solution steps into a MathSolution object.
        
        Why here: This method standardizes the solution format, ensuring consistency
        in how solutions are represented and making it easier to process and evaluate them.
        """
        final_step = steps[-1]
        numerical_answer = self._extract_numerical_answer(" ".join(step.step for step in steps))
        return MathSolution(steps=steps, final_answer=final_step.step, numerical_answer=numerical_answer)

    def _extract_numerical_answer(self, text: str) -> Optional[float]:
        """
        Extracts the numerical answer from the solution text.
        
        Why here: Accurate extraction of the numerical answer is crucial for evaluating
        the correctness of solutions. This method uses regex patterns to identify
        numerical answers in various formats, improving the robustness of the solution evaluation.
        """
        # First, look for patterns like "x = 7" or "The answer is 42 km/h" at the end of the text
        match = re.search(r'(?:x\s*=\s*|answer\s*is\s*|equals\s*|result\s*is\s*)(\d+(?:\.\d+)?)\s*(?:km/h|m/s|mph)?\s*$', text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        
        # If not found at the end, search throughout the text
        match = re.search(r'(?:x\s*=\s*|answer\s*is\s*|equals\s*|result\s*is\s*)(\d+(?:\.\d+)?)\s*(?:km/h|m/s|mph)?', text, re.IGNORECASE)
        if match:
            return float(match.group(1))
        
        # If still not found, look for the last number in the text
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if numbers:
            return float(numbers[-1])
        
        return None

    def _verify_and_correct_solution(self, question: str, solution: MathSolution) -> MathSolution:
        """
        Verifies the solution and corrects it if necessary.
        
        Why here: This final verification step is crucial for ensuring the accuracy
        of the solution. It leverages the LLM's capability to check the solution's
        correctness and make necessary corrections, adding an extra layer of reliability
        to the Q* algorithm's output.
        """
        print("QStar: Verifying and correcting solution")
        verification_prompt = f"""
        Question: {question}
        
        Proposed solution steps:
        {' '.join(step.step for step in solution.steps)}
        
        Proposed final answer: {solution.final_answer}
        Extracted numerical answer: {solution.numerical_answer}
        
        Please verify if the solution and numerical answer are correct and in the requested units. If not, provide the correct numerical answer and explain the correction. If possible, perform a simple calculation to check the answer.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert math problem solver and verifier. Ensure the answer is correct and in the requested units."},
                {"role": "user", "content": verification_prompt}
            ]
        )
        
        verification_result = response.choices[0].message.content.strip()
        
        if "correct" in verification_result.lower():
            print("QStar: Solution verified as correct")
            return solution
        else:
            print("QStar: Solution needs correction")
            corrected_answer = self._extract_numerical_answer(verification_result)
            if corrected_answer is not None:
                solution.numerical_answer = corrected_answer
                solution.final_answer = f"The correct answer is {corrected_answer}"
                solution.steps.append(MathStep(step=f"Correction: {corrected_answer}", explanation=verification_result))
            return solution

def main():
    """
    Main function to demonstrate the Q* algorithm on sample math problems.
    
    Why here: This function serves as an entry point and demonstration of how
    the Q* algorithm can be applied to various math problems. It showcases the
    versatility of the approach across different problem types.
    """
    solver = MathProblemSolver()
    q_estimator = QValueEstimator()
    q_star = QStar(solver, q_estimator)

    problems = [
        "If a train travels 120 km in 2 hours, what is its average speed in km/h?",
        "A rectangle has a length of 8 cm and a width of 5 cm. What is its area?",
        "If x + 3 = 10, what is the value of x?",
        "A store offers a 20% discount on a $50 item. What is the final price?",
        "If 3x - 7 = 14, solve for x."
    ]

    for i, problem in enumerate(problems, 1):
        print(f"\nSolving Problem {i}: {problem}")
        solution = q_star.solve(problem)
        print(f"\nProblem {i}: {problem}")
        print("Solution:")
        for step in solution.steps:
            print(f"- {step.step}")
            print(f"  Explanation: {step.explanation}")
        print(f"Final Answer: {solution.final_answer}")
        if solution.numerical_answer is not None:
            print(f"Numerical Answer: {solution.numerical_answer}")
        print("-" * 50)

if __name__ == "__main__":
    main()

"""
Conclusion and Key Takeaways:

1. MDP Formulation: The Q* algorithm formulates multi-step reasoning as a Markov
   Decision Process, allowing for a structured approach to problem-solving.

2. Heuristic Search: By using Q-values as a heuristic, the algorithm efficiently
   navigates the solution space, prioritizing promising paths.

3. LLM Integration: The implementation leverages LLMs for multiple aspects of
   problem-solving, including step generation, Q-value estimation, and solution
   verification.

4. Beam Search: The use of beam search allows the algorithm to explore multiple
   solution paths simultaneously, increasing the chances of finding optimal solutions.

5. Verification Step: The final verification and correction step adds an extra
   layer of reliability to the solutions produced by the algorithm.

6. Flexibility: This implementation demonstrates how the Q* approach can be applied
   to various types of math problems, showcasing its potential for broader applications
   in multi-step reasoning tasks.

7. Interpretability: By breaking down the problem-solving process into discrete steps
   and providing explanations, the Q* algorithm offers a more interpretable approach
   compared to end-to-end black-box models.

Future Directions:
- Extend the implementation to other domains beyond math problem solving.
- Explore more sophisticated Q-value estimation techniques, possibly incorporating
  reinforcement learning approaches.
- Investigate ways to optimize the algorithm for better efficiency, especially for
  more complex, multi-step reasoning tasks.
- Develop methods to incorporate domain-specific knowledge or constraints into the
  Q* framework for specialized applications.

This implementation serves as a proof of concept for the Q* algorithm, demonstrating
its potential to enhance the multi-step reasoning capabilities of large language models.
"""