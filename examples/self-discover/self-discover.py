import os
import json
from typing import List, Dict, Any
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class SelfDiscover:
    def __init__(self, model="gpt-4o"):
        self.model = model

    def generate_completion(self, messages: List[Dict[str, str]], expected_structure: Dict[str, Any]) -> Any:
        """
        Generate a completion using the OpenAI API with structured output.
        
        This method encapsulates the API call and error handling, ensuring
        that the output adheres to the expected structure.
        """
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                functions=[{
                    "name": "process_response",
                    "description": "Process the response according to the expected structure",
                    "parameters": expected_structure
                }],
                function_call={"name": "process_response"}
            )
            return json.loads(response.choices[0].message.function_call.arguments)
        except Exception as e:
            print(f"Error in generate_completion: {e}")
            return {}

    def select_modules(self, task: str) -> List[str]:
        """
        SELECT step: Choose relevant reasoning modules for the given task.
        
        This method demonstrates the first step of SELF-DISCOVER, where
        the model selects appropriate reasoning modules based on the task.
        """
        messages = [
            {"role": "system", "content": "You are an expert in selecting appropriate reasoning modules for problem-solving."},
            {"role": "user", "content": f"Select the most relevant reasoning modules for the following task: '{task}'. Return only the list of module names."}
        ]
        
        expected_structure = {
            "type": "object",
            "properties": {
                "modules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of selected reasoning modules"
                }
            },
            "required": ["modules"]
        }
        
        result = self.generate_completion(messages, expected_structure)
        return result.get("modules", [])

    def adapt_modules(self, task: str, selected_modules: List[str]) -> List[str]:
        """
        ADAPT step: Tailor the selected modules to the specific task.
        
        This method represents the second step of SELF-DISCOVER, where
        the model adapts the selected modules to be more specific to the given task.
        """
        modules_str = ", ".join(selected_modules)
        messages = [
            {"role": "system", "content": "You are an expert in adapting reasoning modules to specific tasks."},
            {"role": "user", "content": f"Adapt the following reasoning modules to the task: '{task}'\nModules: {modules_str}. Return only the list of adapted module descriptions."}
        ]
        
        expected_structure = {
            "type": "object",
            "properties": {
                "adapted_modules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of adapted reasoning modules"
                }
            },
            "required": ["adapted_modules"]
        }
        
        result = self.generate_completion(messages, expected_structure)
        return result.get("adapted_modules", [])

    def implement_structure(self, task: str, adapted_modules: List[str]) -> Dict[str, str]:
        """
        IMPLEMENT step: Create a reasoning structure based on adapted modules.
        
        This method demonstrates the third step of SELF-DISCOVER, where
        the model generates a step-by-step reasoning structure for solving the task.
        """
        modules_str = ", ".join(adapted_modules)
        messages = [
            {"role": "system", "content": "You are an expert in creating structured reasoning plans for problem-solving."},
            {"role": "user", "content": f"Create a step-by-step reasoning structure for the task: '{task}', using these adapted modules:\n{modules_str}. Return a JSON object with numbered steps as keys and empty strings as values."}
        ]
        
        expected_structure = {
            "type": "object",
            "properties": {
                "structure": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Step-by-step reasoning structure with steps as keys and empty strings as values"
                }
            },
            "required": ["structure"]
        }
        
        result = self.generate_completion(messages, expected_structure)
        structure = result.get("structure", {})
        if not structure:
            print("Warning: Empty reasoning structure received. Creating a structure based on adapted modules.")
            structure = {}
            for i, module in enumerate(adapted_modules, 1):
                structure[f"Step {i}"] = f"Apply {module.split(':')[0].strip()}: "
            structure["Final Step"] = "Synthesize findings and conclude: "
        return structure

    def solve_task(self, task: str, structure: Dict[str, str]) -> Dict[str, Any]:
        """
        Solve the task using the generated reasoning structure.
        
        This method applies the reasoning structure to solve the given task,
        demonstrating how SELF-DISCOVER uses the generated structure for problem-solving.
        """
        structure_str = json.dumps(structure)
        messages = [
            {"role": "system", "content": "You are an expert problem solver. Follow the given reasoning structure to solve the task step by step."},
            {"role": "user", "content": f"Solve the following task by filling in each step of the reasoning structure:\nTask: {task}\nReasoning structure: {structure_str}\n\nProvide the solution by filling in each step and then give a final answer that summarizes all steps."}
        ]
        
        expected_structure = {
            "type": "object",
            "properties": {
                "solution": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Filled reasoning structure with solution steps"
                },
                "final_answer": {"type": "string", "description": "A summary of all steps and the final answer to the task"}
            },
            "required": ["solution", "final_answer"]
        }
        
        result = self.generate_completion(messages, expected_structure)
        
        # Ensure the final_answer contains all steps
        if result and "solution" in result and "final_answer" in result:
            steps_summary = "\n".join([f"{step}: {content}" for step, content in result["solution"].items()])
            result["final_answer"] = f"{steps_summary}\n\nFinal Answer: {result['final_answer']}"
        
        return result

def run_self_discover(task: str) -> Dict[str, Any]:
    """
    Run the complete SELF-DISCOVER process for a given task.
    
    This function orchestrates the entire SELF-DISCOVER approach,
    from selecting modules to solving the task using the generated structure.
    """
    sd = SelfDiscover()
    
    print(f"Task: {task}")
    
    try:
        selected_modules = sd.select_modules(task)
        print(f"Selected modules: {selected_modules}")
        
        adapted_modules = sd.adapt_modules(task, selected_modules)
        print(f"Adapted modules: {adapted_modules}")
        
        structure = sd.implement_structure(task, adapted_modules)
        print(f"Reasoning structure: {json.dumps(structure, indent=2)}")
        
        solution = sd.solve_task(task, structure)
        print(f"Solution:\n{json.dumps(solution, indent=2)}")
        
        return {
            "task": task,
            "selected_modules": selected_modules,
            "adapted_modules": adapted_modules,
            "reasoning_structure": structure,
            "solution": solution
        }
    except Exception as e:
        print(f"Error in run_self_discover: {e}")
        return {
            "task": task,
            "error": str(e)
        }

# Example usage with diverse tasks
tasks = [
    "Calculate the area of a triangle with base 6 cm and height 8 cm.",
    "Explain why the sky appears blue.",
    "Solve the equation: 2x + 5 = 13",
    "A rectangular garden is 3 meters longer than it is wide. If the perimeter of the garden is 26 meters, what are its dimensions?",
    "In a group of 5 friends, if Alice is taller than Bob, Bob is taller than Charlie, Charlie is shorter than David, and David is shorter than Eve, who is the tallest and who is the shortest?",
    "A ball is thrown vertically upward with an initial velocity of 20 m/s from a height of 1.5 m above the ground. How long does it take for the ball to reach its maximum height, and what is that height? (Assume g = 9.8 m/s^2)",
    "Design a system to automatically water plants in a garden based on soil moisture levels and weather forecasts.",
    "A self-driving car is about to crash. It can either swerve left and hit a group of elderly pedestrians, or swerve right and hit a group of young children. What ethical considerations should guide its decision?",
    "Explain the process of photosynthesis and its importance for life on Earth.",
    "Analyze the causes and consequences of the Industrial Revolution.",
    "Write a haiku about the changing seasons.",
    "Estimate the number of piano tuners in Chicago.",
    "Describe how machine learning algorithms can be used to detect fraud in financial transactions."
]

results = []
for task in tasks:
    print("\n" + "="*50)
    result = run_self_discover(task)
    results.append(result)
    print("="*50)

# Save results to a JSON file
with open("self_discover_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to self_discover_results.json")