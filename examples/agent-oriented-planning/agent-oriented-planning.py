import os
import json
from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Pydantic models for structured output
# These models ensure type safety and validation of our data structures
class SubTask(BaseModel):
    task: str
    id: int
    name: str
    reason: str
    dep: List[int] = Field(default_factory=list)

class Plan(BaseModel):
    sub_tasks: List[SubTask]

class RewardModelOutput(BaseModel):
    score: float = 0.0

class AgentResponse(BaseModel):
    content: str
    score: float = 0.0

# Meta-agent for task decomposition and allocation
def meta_agent(query: str) -> Plan:
    """
    This function represents the meta-agent that decomposes a given query into sub-tasks
    and allocates them to specific agents. It's a critical component of the agent-oriented
    planning system, embodying the paper's concept of task decomposition.
    """
    prompt = f"""
    You are a planning agent responsible for decomposing the given query into sub-tasks
    and choosing the most suitable agent for each sub-task. Your goal is to efficiently
    and accurately complete task planning based on the provided agent descriptions,
    ensuring the coherence and quality of the sub-tasks.

    Please output the sub-tasks and corresponding agents in the following JSON format:
    {{
        "sub_tasks": [
            {{
                "task": "Description of the sub-task",
                "id": 1,
                "name": "name_of_agent",
                "reason": "Detailed reason for choosing this agent",
                "dep": [0]
            }},
            // ... more sub-tasks ...
        ]
    }}

    IMPORTANT:
    - 'id' must be a unique integer for each sub-task, starting from 1.
    - Elements in 'dep' must be integers referring to the 'id' of prerequisite tasks.
    - Use 0 in 'dep' if there are no dependencies.

    Available agents and their descriptions:
    - code_agent: Generate code in Python for precise computations.
    - math_agent: Answer math questions by reasoning step-by-step.
    - search_agent: Call Bing Search API for obtaining information.
    - commonsense_agent: Answer questions using commonsense reasoning.

    Given the user query, output the task plan in the specified JSON format. Ensure all
    important information such as nouns or numbers from the query are included in the sub-tasks.

    User query: {query}
    """

    # Use the OpenAI API to generate the plan
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    # Parse the JSON response into our Plan model
    return Plan.model_validate_json(response.choices[0].message.content)

# Reward model for evaluating sub-task solvability
def reward_model(sub_task: SubTask) -> float:
    """
    This function implements the reward model concept from the paper.
    It evaluates the solvability of a sub-task by a given agent,
    providing a score that guides further processing of the sub-task.
    """
    prompt = f"""
    Evaluate the solvability of the following sub-task by the assigned agent. 
    Provide a score between 0 and 1, where 1 means highly solvable and 0 means unsolvable.
    Return the score in the following JSON format:
    {{
        "score": 0.75
    }}

    Sub-task: {sub_task.task}
    Assigned agent: {sub_task.name}

    JSON Output:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    try:
        reward_output = RewardModelOutput.model_validate_json(response.choices[0].message.content)
        return reward_output.score
    except Exception as e:
        print(f"Error parsing reward model output: {e}")
        print(f"Raw output: {response.choices[0].message.content}")
        return 0.0

# Specialized agents
# These functions represent the different types of agents mentioned in the paper,
# each specialized for a specific type of task.

def code_agent(task: str) -> AgentResponse:
    """Generate Python code to solve the given task."""
    prompt = f"""
    You are a code agent. Generate Python code to solve the given task.
    Task: {task}
    Code:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return AgentResponse(content=response.choices[0].message.content)

def math_agent(task: str) -> AgentResponse:
    """Solve math problems step-by-step."""
    prompt = f"""
    You are a math agent. Solve the given math problem step-by-step.
    Task: {task}
    Solution:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return AgentResponse(content=response.choices[0].message.content)

def search_agent(task: str) -> AgentResponse:
    """Simulate searching for information."""
    prompt = f"""
    You are a search agent. Provide information related to the given task.
    Task: {task}
    Information:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return AgentResponse(content=response.choices[0].message.content)

def commonsense_agent(task: str) -> AgentResponse:
    """Answer questions using common sense reasoning."""
    prompt = f"""
    You are a commonsense agent. Answer the given question using common sense reasoning.
    Task: {task}
    Answer:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return AgentResponse(content=response.choices[0].message.content)

# Task modification functions
# These functions implement the paper's concepts of task modification,
# allowing for replanning, detailed planning, and task re-description.

def replan(sub_task: SubTask) -> SubTask:
    """Replan a sub-task that cannot be solved by any agent."""
    prompt = f"""
    The following sub-task cannot be solved by any agent. Replan this sub-task:
    {sub_task.task}

    New sub-task:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return SubTask(
        task=response.choices[0].message.content,
        id=sub_task.id,
        name=sub_task.name,
        reason="Replanned task",
        dep=sub_task.dep
    )

def plan_in_detail(sub_task: SubTask) -> List[SubTask]:
    """Break down a complex sub-task into smaller, more manageable sub-tasks."""
    prompt = f"""
    The following sub-task is too complex and needs to be broken down further:
    {sub_task.task}

    Provide a more detailed plan with smaller sub-tasks using the following JSON format:
    {{
        "sub_tasks": [
            {{
                "task": "Description of the smaller sub-task",
                "id": 1,
                "name": "name_of_agent",
                "reason": "Detailed reason for choosing this agent",
                "dep": [0]
            }},
            // ... more sub-tasks ...
        ]
    }}

    IMPORTANT:
    - 'id' must be a unique integer for each sub-task, starting from 1.
    - Elements in 'dep' must be integers referring to the 'id' of prerequisite tasks.
    - Use 0 in 'dep' if there are no dependencies.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )

    detailed_plan = Plan.model_validate_json(response.choices[0].message.content)
    return detailed_plan.sub_tasks

def re_describe(sub_task: SubTask) -> SubTask:
    """Make a sub-task clearer and more specific."""
    prompt = f"""
    Rewrite the following sub-task to make it clearer and more specific:
    {sub_task.task}

    Rewritten sub-task:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return SubTask(
        task=response.choices[0].message.content,
        id=sub_task.id,
        name=sub_task.name,
        reason="Re-described task",
        dep=sub_task.dep
    )

# Detector for ensuring completeness and non-redundancy
def detect_issues(query: str, plan: Plan) -> str:
    """
    This function implements the paper's concept of ensuring completeness
    and non-redundancy in the task decomposition.
    """
    prompt = f"""
    Analyze the following plan for completeness and non-redundancy:

    Original query: {query}

    Plan:
    {json.dumps(plan.model_dump(), indent=2)}

    Identify any missing information or redundant tasks:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# Main function to orchestrate the process
def solve_query(query: str) -> str:
    """
    This function orchestrates the entire agent-oriented planning process,
    implementing the workflow described in the paper.
    """
    # Step 1: Meta-agent decomposes the query
    plan = meta_agent(query)

    # Step 2: Evaluate and modify sub-tasks
    final_plan = Plan(sub_tasks=[])
    for sub_task in plan.sub_tasks:
        score = reward_model(sub_task)
        print(f"Sub-task: {sub_task.task}")
        print(f"Score: {score}")
        if score < 0.3:
            sub_task = replan(sub_task)
        elif score < 0.7:
            detailed_tasks = plan_in_detail(sub_task)
            final_plan.sub_tasks.extend(detailed_tasks)
            continue
        elif score < 0.9:
            sub_task = re_describe(sub_task)
        final_plan.sub_tasks.append(sub_task)

    # Step 3: Detect and address issues
    issues = detect_issues(query, final_plan)
    if issues.strip():
        print(f"Issues detected: {issues}")
        # Here you could implement logic to address the issues

    # Step 4: Execute the plan
    results = []
    for sub_task in final_plan.sub_tasks:
        if sub_task.name == "code_agent":
            result = code_agent(sub_task.task)
        elif sub_task.name == "math_agent":
            result = math_agent(sub_task.task)
        elif sub_task.name == "search_agent":
            result = search_agent(sub_task.task)
        elif sub_task.name == "commonsense_agent":
            result = commonsense_agent(sub_task.task)
        results.append(result)

    # Step 5: Combine results (simplified for this example)
    final_answer = "\n".join([f"Sub-task {i+1}: {r.content}" for i, r in enumerate(results)])

    return final_answer

# Example usage
if __name__ == "__main__":
    queries = [
        "What is the population difference between New York City and Los Angeles?",
        "Calculate the compound interest on $10,000 invested for 5 years at 5% annual interest rate.",
        "How many Olympic-sized swimming pools could be filled with the daily water consumption of Tokyo?",
        "What is the carbon footprint difference between driving a gasoline car and an electric car for 10,000 miles?",
        "If the average person blinks 15 times per minute, how many times would they blink during a full day of watching all Star Wars movies back-to-back?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        answer = solve_query(query)
        print(f"Answer:\n{answer}")
        print("-" * 50)

"""
Key Ideas and Future Improvements:

1. Agent-Oriented Planning: The system implements the paper's core concept of breaking down
   complex tasks into sub-tasks and assigning them to specialized agents.

2. Meta-Agent: The meta_agent function embodies the idea of a central entity that can
   understand and decompose queries, addressing the challenge of automatic task decomposition.

3. Reward Model: The reward_model function implements the paper's concept of evaluating
   sub-task solvability without actual execution, improving efficiency.

4. Task Modification: The replan, plan_in_detail, and re_describe functions implement
   the paper's ideas on how to handle tasks that are unsolvable, too complex, or unclear.

5. Completeness and Non-Redundancy: The detect_issues function addresses the paper's
   emphasis on ensuring that the decomposed tasks cover all aspects of the original query
   without redundancy.

6. Structured Output: The use of Pydantic models ensures that the system's internal
   data structures are well-defined and validated.

Future Improvements:

1. Feedback Loop: Implement a more robust feedback mechanism to continuously improve
   the meta-agent's task decomposition and allocation skills.

2. Dynamic Agent Capabilities: Allow for dynamic updating of agent capabilities,
   possibly through a learning mechanism.

3. More Sophisticated Issue Resolution: Enhance the system's ability to automatically
   address issues detected in the plan, rather than just reporting them.

4. Parallel Execution: Implement parallel execution of independent sub-tasks to
   improve overall efficiency.

5. Real-World Integration: Integrate with actual external tools and APIs, especially
   for the search_agent, to handle real-world data and queries.

6. User Interaction: Add a mechanism for user feedback and interaction during the
   planning and execution process.

7. Performance Metrics: Implement more comprehensive metrics to evaluate the system's
   performance, including execution time, accuracy, and resource utilization.
"""