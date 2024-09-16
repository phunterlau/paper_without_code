import os
import json
from typing import List, Optional, Dict, Tuple
from pydantic import BaseModel
from openai import OpenAI

# Set up OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define data structures for workflows and steps
class WorkflowStep(BaseModel):
    """
    Represents a single step in a workflow.
    Each step contains the environment state, reasoning, and action to be taken.
    This aligns with the paper's emphasis on capturing context and decision-making process.
    """
    environment_state: str  # Current state of the environment
    reasoning: str  # Explanation for the action
    action_type: str  # Type of action (e.g., CLICK, TYPE)
    target: str  # Target of the action (e.g., button id, input field)
    value: Optional[str] = None  # Optional value for the action (e.g., text to type)

class Workflow(BaseModel):
    """
    Represents a complete workflow.
    This structure allows for abstract, reusable workflows as described in the paper.
    """
    name: str
    description: str
    domain: str  # Domain helps in cross-domain generalization
    steps: List[WorkflowStep]

class WorkflowList(BaseModel):
    """
    A list of workflows, used for batch processing of multiple workflows.
    """
    workflows: List[Workflow]

class AWM:
    """
    Agent Workflow Memory (AWM) class.
    This class implements the core functionality of the AWM system as described in the paper.
    """
    def __init__(self):
        self.workflows: List[Workflow] = []  # Offline induced workflows
        self.online_workflows: List[Workflow] = []  # Online induced workflows

    def induce_workflow_offline(self, experiences: str):
        """
        Induce workflows from a set of experiences in an offline setting.
        This method aligns with the paper's description of offline workflow induction,
        where workflows are extracted from pre-existing annotated examples.
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in extracting common workflows from user experiences. Focus on abstracting example-specific contexts and identifying reusable sub-routines. Use placeholders like {product-type}, {product-color}, etc."},
                {"role": "user", "content": f"Extract common workflows from the given experiences, focusing on reusable sub-tasks and using abstract placeholders:\n{experiences}"}
            ],
            functions=[{
                "name": "extract_workflows",
                "description": "Extract common workflows from given experiences",
                "parameters": WorkflowList.model_json_schema()
            }],
            function_call={"name": "extract_workflows"}
        )
        
        result = WorkflowList.model_validate_json(response.choices[0].message.function_call.arguments)
        self.workflows.extend(result.workflows)

    def induce_workflow_online(self, experience: str):
        """
        Induce a workflow from a single experience in an online setting.
        This method implements the online scenario described in the paper,
        where workflows are generated in real-time from agent actions.
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in extracting reusable workflows from individual experiences. Focus on identifying sub-routines that could be applied to similar tasks. Use abstract placeholders where possible."},
                {"role": "user", "content": f"Extract a reusable workflow from this experience, using abstract placeholders:\n{experience}"}
            ],
            functions=[{
                "name": "extract_workflow",
                "description": "Extract a reusable workflow from the given experience",
                "parameters": Workflow.model_json_schema()
            }],
            function_call={"name": "extract_workflow"}
        )
        
        result = Workflow.model_validate_json(response.choices[0].message.function_call.arguments)
        self.online_workflows.append(result)

    def apply_workflow(self, task: str, environment_state: str, domain: str) -> Tuple[List[WorkflowStep], str]:
        """
        Apply the most relevant workflow to a given task and current environment state.
        This method demonstrates the system's ability to adapt workflows to new tasks,
        potentially across different domains, as emphasized in the paper.
        """
        all_workflows = self.workflows + self.online_workflows
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in applying workflows to specific tasks, considering the current environment state and domain. Adapt workflows from other domains if necessary."},
                {"role": "user", "content": f"Apply the most relevant workflow to this task: {task}\nCurrent environment state: {environment_state}\nDomain: {domain}\nAvailable workflows: {json.dumps([w.model_dump() for w in all_workflows])}"}
            ],
            functions=[{
                "name": "apply_workflow",
                "description": "Apply the most relevant workflow to the given task and environment state",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": WorkflowStep.model_json_schema()
                        },
                        "workflow_used": {"type": "string"}
                    },
                    "required": ["steps", "workflow_used"]
                }
            }],
            function_call={"name": "apply_workflow"}
        )
        
        result = json.loads(response.choices[0].message.function_call.arguments)
        return [WorkflowStep.model_validate(s) for s in result['steps']], result['workflow_used']

    def evaluate_success(self, task: str, steps: List[WorkflowStep]) -> bool:
        """
        Evaluate if the given steps successfully solve the task.
        This method implements the success evaluation mechanism described in the paper,
        which is crucial for determining when to induce new workflows.
        """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in evaluating the success of web navigation tasks."},
                {"role": "user", "content": f"Evaluate if these steps successfully solve the task:\nTask: {task}\nSteps: {json.dumps([s.model_dump() for s in steps])}"}
            ],
            functions=[{
                "name": "evaluate_success",
                "description": "Evaluate if the steps successfully solve the task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "reason": {"type": "string"}
                    },
                    "required": ["success", "reason"]
                }
            }],
            function_call={"name": "evaluate_success"}
        )
        
        result = json.loads(response.choices[0].message.function_call.arguments)
        return result['success']
    
    def execute_workflow_action(self, action_name: str, params: Dict):
        """
        Execute a high-level workflow action.
        This method demonstrates how workflows can be used as high-level actions,
        expanding the agent's action space as described in the paper.
        """
        workflow = next((w for w in self.workflows + self.online_workflows if w.name == action_name), None)
        if workflow:
            print(f"Executing workflow action: {action_name}")
            for step in workflow.steps:
                print(f"  {step.action_type}('{step.target}', {step.value})")
        else:
            print(f"Workflow action {action_name} not found")

    def get_total_workflows(self) -> int:
        """
        Get the total number of workflows (offline + online).
        This method helps track the growth of the workflow library over time.
        """
        return len(self.workflows) + len(self.online_workflows)

def demonstrate_snowball_effect(awm: AWM):
    """
    Demonstrate the snowball effect using the AWM system.
    This function showcases how the system builds increasingly complex workflows over time,
    a key concept described in the paper.
    """
    print("Demonstrating the snowball effect of learning increasingly complex workflows:\n")

    tasks = [
        ("Search for a laptop on an e-commerce website", "On the homepage of an e-commerce website", "e-commerce"),
        ("Add a laptop to the shopping cart", "On the search results page for laptops", "e-commerce"),
        ("Complete the purchase of a laptop in the shopping cart", "On the shopping cart page", "e-commerce"),
        ("Search for, add to cart, and purchase a laptop", "On the homepage of an e-commerce website", "e-commerce"),
        ("Book a flight ticket", "On the homepage of a travel website", "travel"),
        ("Plan a business trip: book a flight, reserve a hotel, and purchase a laptop for the trip", "On a multi-purpose travel and shopping website", "travel,e-commerce")
    ]

    for i, (task, env_state, domain) in enumerate(tasks, 1):
        print(f"Task {i}: {task}")
        steps, workflow_used = awm.apply_workflow(task, env_state, domain)
        print(f"Using workflow: {workflow_used}")
        print("Steps:")
        for step in steps:
            print(f"  Action: {step.action_type}('{step.target}'{', ' + repr(step.value) if step.value else ''})")
        
        if awm.evaluate_success(task, steps):
            awm.induce_workflow_online(json.dumps({"task": task, "steps": [s.model_dump() for s in steps], "domain": domain}))
            print(f"Task {i} completed successfully. New workflow induced.")
        else:
            print(f"Task {i} not completed successfully.")
        
        print(f"Total workflows: {awm.get_total_workflows()}\n")

# Example usage
if __name__ == "__main__":
    awm = AWM()

    # Induce initial offline workflow
    # This demonstrates the offline workflow induction process
    shopping_experience = """
    Task: Buy a {product-type}
    Actions:
    1. Environment: On the homepage of an e-commerce website
       Reasoning: To find {product-type}, I need to use the search function
       Action: TYPE('search-bar', '{product-type}')
    2. Environment: Search results for {product-type} are displayed
       Reasoning: I need to select a product to purchase
       Action: CLICK('product-1')
    3. Environment: On the product page for a {product-type}
       Reasoning: I want to add this item to my cart
       Action: CLICK('add-to-cart')
    4. Environment: Item added to cart, popup appears
       Reasoning: I want to complete my purchase
       Action: CLICK('checkout')
    5. Environment: On the checkout page
       Reasoning: I need to confirm my purchase
       Action: CLICK('confirm-purchase')
    """

    awm.induce_workflow_offline(shopping_experience)
    print("Induced offline shopping workflows:")
    for workflow in awm.workflows:
        print(f"Name: {workflow.name}")
        print(f"Description: {workflow.description}")
        print(f"Domain: {workflow.domain}")
        print("Steps:")
        for step in workflow.steps:
            print(f"  Environment: {step.environment_state}")
            print(f"  Reasoning: {step.reasoning}")
            print(f"  Action: {step.action_type}('{step.target}'{', ' + repr(step.value) if step.value else ''})")
        print()

    # Cross-domain task: Apply shopping workflow to travel booking
    # This demonstrates the system's ability to adapt workflows across domains
    travel_task = "Book a flight from {origin-city} to {destination-city} for next month"
    travel_steps, workflow_used = awm.apply_workflow(travel_task, "On the homepage of a travel booking website", "travel")
    print(f"Applied steps for '{travel_task}' using workflow: {workflow_used}")
    for step in travel_steps:
        print(f"Environment: {step.environment_state}")
        print(f"Reasoning: {step.reasoning}")
        print(f"Action: {step.action_type}('{step.target}'{', ' + repr(step.value) if step.value else ''})")
        print()

    # Evaluate success and induce new workflow if successful
    # This shows the online workflow induction process
    if awm.evaluate_success(travel_task, travel_steps):
        print("Task completed successfully. Inducing new workflow.")
        awm.induce_workflow_online(json.dumps({"task": travel_task, "steps": [s.model_dump() for s in travel_steps], "domain": "travel"}))
    else:
        print("Task not completed successfully.")

    # Demonstrate workflow action execution
    # This shows how workflows can be used as high-level actions
    print("Executing high-level workflow action:")
    awm.execute_workflow_action("Purchase {product-color} {product-type}", {"product-color": "black", "product-type": "laptop"})

    # Cross-website task: Apply travel booking workflow to a different travel website
    # This demonstrates generalization across websites within the same domain
    cross_website_task = "Book a hotel in {destination-city} for next month"
    hotel_steps, workflow_used = awm.apply_workflow(cross_website_task, "On the homepage of a hotel booking website", "travel")
    print(f"\nApplied steps for cross-website task '{cross_website_task}' using workflow: {workflow_used}")
    for step in hotel_steps:
        print(f"Environment: {step.environment_state}")
        print(f"Reasoning: {step.reasoning}")
        print(f"Action: {step.action_type}('{step.target}'{', ' + repr(step.value) if step.value else ''})")
        print()

    # Demonstrate the "snowball effect" of building increasingly complex workflows
    # This showcases how the system can handle more complex, multi-step tasks over time
    complex_task = "Plan a vacation: book flights, reserve a hotel, and rent a car"
    complex_steps, workflow_used = awm.apply_workflow(complex_task, "On a travel planning website homepage", "travel")
    print(f"\nApplied steps for complex task '{complex_task}' using workflow: {workflow_used}")
    for step in complex_steps:
        print(f"Environment: {step.environment_state}")
        print(f"Reasoning: {step.reasoning}")
        print(f"Action: {step.action_type}('{step.target}'{', ' + repr(step.value) if step.value else ''})")
        print()

    if awm.evaluate_success(complex_task, complex_steps):
        print("Complex task completed successfully. Inducing new, more complex workflow.")
        awm.induce_workflow_online(json.dumps({"task": complex_task, "steps": [s.model_dump() for s in complex_steps], "domain": "travel"}))
        print(f"Total workflows after induction: {len(awm.workflows) + len(awm.online_workflows)}")
    else:
        print("Complex task not completed successfully.")

    # Demonstrate the snowball effect
    # This function provides a clear illustration of how the system builds up its capabilities over time
    demonstrate_snowball_effect(awm)

"""
Agent Workflow Memory (AWM) Implementation Review

This implementation satisfies the core ideas presented in the AWM paper:

Strengths:
1. Abstract Workflow Representation: Uses placeholders for better generalization.
2. Cross-Domain Application: Demonstrates applying workflows across domains.
3. Workflow Induction: Implements both offline and online induction.
4. Snowball Effect: Shows progression from simple to complex tasks.
5. Adaptability: Adapts workflows to new scenarios.
6. Environment State and Reasoning: Captures context and decision-making processes.
7. Continuous Learning: Demonstrates increasing number of workflows over time.

Areas for Improvement:
1. Workflow Combination: Could more clearly show how complex workflows are built from simpler ones.
2. Error Handling and Adaptation: Doesn't demonstrate handling of imperfect workflow matches.
3. Sub-routine Reuse: Could more explicitly show reuse of common sub-routines across tasks.
4. Workflow Actions: Fails to execute high-level workflow actions as described in the paper.
5. Quantitative Evaluation: Lacks concrete metrics on task success rates and efficiency improvements.
"""