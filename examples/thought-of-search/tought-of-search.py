import os
import json
from typing import List, Dict, Any
from pydantic import BaseModel
from openai import OpenAI
import heapq

# Initialize OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class SearchComponents(BaseModel):
    successor_function: str
    goal_test: str
    heuristic_function: str = ""

class Problem(BaseModel):
    name: str
    description: str
    use_astar: bool = False
    use_dfs: bool = False

def generate_search_components(problem: Problem) -> SearchComponents:
    """
    Generate search components (successor function, goal test, and optionally heuristic function)
    for a given problem using GPT-4-Turbo.
    """
    heuristic_prompt = """
    3. A heuristic function named 'heuristic_function' that takes a state and returns an estimate of the cost to reach the goal state.
    The heuristic should be admissible (never overestimate the cost) and consistent.
    """ if problem.use_astar else ""

    prompt = f"""
    Given the following search problem:
    Name: {problem.name}
    Description: {problem.description}

    Please generate Python code for:
    1. A successor function named 'successor_function' that takes a state and returns a list of valid successor states.
    2. A goal test function named 'goal_test' that takes a state and returns True if it's a goal state, False otherwise.
    {heuristic_prompt}

    Ensure the functions are efficient and correctly handle the problem constraints.
    For the 8-Puzzle, use the Manhattan distance as the heuristic.
    For Tower of Hanoi, implement a heuristic that counts the number of disks not in their final position.
    For N-Queens, implement a successor function that places queens in valid positions only, and a goal test that checks if all queens are placed without conflicts.
    For Game 24, implement a successor function that applies arithmetic operations to pairs of numbers, and a goal test that checks if the result is 24.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert AI assistant specializing in search algorithms and problem-solving."},
            {"role": "user", "content": prompt}
        ],
        tools=[{
            "type": "function",
            "function": {
                "name": "provide_search_components",
                "description": "Provide successor function, goal test, and optionally heuristic function for a search problem",
                "parameters": SearchComponents.model_json_schema()
            }
        }],
        tool_choice={"type": "function", "function": {"name": "provide_search_components"}}
    )

    return SearchComponents(**json.loads(response.choices[0].message.tool_calls[0].function.arguments))

def solve_problem(problem: Problem, initial_state: Any, max_iterations: int = 1000000) -> List[Any]:
    """
    Solve the given problem using the appropriate search algorithm.
    """
    # Generate search components using the LLM
    components = generate_search_components(problem)
    
    # Create a new namespace and execute the generated functions in it
    namespace = {}
    exec(components.successor_function, namespace)
    exec(components.goal_test, namespace)
    if problem.use_astar:
        exec(components.heuristic_function, namespace)
    
    # Retrieve the functions from the namespace
    successor_function = namespace['successor_function']
    goal_test = namespace['goal_test']
    heuristic_function = namespace.get('heuristic_function')
    
    # Choose the appropriate search algorithm based on the problem configuration
    if problem.use_astar:
        return astar_search(initial_state, successor_function, goal_test, heuristic_function, max_iterations)
    elif problem.use_dfs:
        return dfs_search(initial_state, successor_function, goal_test, max_iterations)
    else:
        return bfs_search(initial_state, successor_function, goal_test, max_iterations)

def bfs_search(initial_state: Any, successor_function, goal_test, max_iterations: int) -> List[Any]:
    """
    Perform Breadth-First Search.
    """
    queue = [(initial_state, [])]
    visited = set()

    for _ in range(max_iterations):
        if not queue:
            return []  # No solution found
        
        state, path = queue.pop(0)
        if goal_test(state):
            return path + [state]  # Solution found
        
        state_str = str(state)
        if state_str not in visited:
            visited.add(state_str)
            for successor in successor_function(state):
                queue.append((successor, path + [state]))
    
    return []  # Max iterations reached, no solution found

def dfs_search(initial_state: Any, successor_function, goal_test, max_iterations: int) -> List[Any]:
    """
    Perform Depth-First Search.
    """
    stack = [(initial_state, [])]
    visited = set()

    for _ in range(max_iterations):
        if not stack:
            return []  # No solution found
        
        state, path = stack.pop()
        if goal_test(state):
            return path + [state]  # Solution found
        
        state_str = str(state)
        if state_str not in visited:
            visited.add(state_str)
            for successor in successor_function(state):
                stack.append((successor, path + [state]))
    
    return []  # Max iterations reached, no solution found

def astar_search(initial_state: Any, successor_function, goal_test, heuristic_function, max_iterations: int) -> List[Any]:
    """
    Perform A* Search.
    """
    frontier = [(0, initial_state, [])]
    visited = set()

    for _ in range(max_iterations):
        if not frontier:
            return []  # No solution found
        
        _, state, path = heapq.heappop(frontier)
        if goal_test(state):
            return path + [state]  # Solution found
        
        state_str = str(state)
        if state_str not in visited:
            visited.add(state_str)
            for successor in successor_function(state):
                new_path = path + [state]
                priority = len(new_path) + heuristic_function(successor)
                heapq.heappush(frontier, (priority, successor, new_path))
    
    return []  # Max iterations reached, no solution found

# Define the problems to be solved
problems = [
    Problem(
        name="8-Puzzle",
        description="Solve the 8-puzzle problem. The state is represented as a list of 9 integers where 0 represents the empty space. The goal is to reach the state [1,2,3,4,5,6,7,8,0].",
        use_astar=True
    ),
    Problem(
        name="Tower of Hanoi",
        description="Solve the Tower of Hanoi problem with 3 pegs and 3 disks. The state is represented as a list of 3 lists, each representing a peg and containing the disks on that peg (smaller numbers represent smaller disks). The goal is to move all disks to the third peg.",
        use_astar=True
    ),
    Problem(
        name="Water Jug",
        description="Solve the Water Jug problem with two jugs of capacity 4 and 3 liters. The state is represented as a tuple (x, y) where x is the amount of water in the 4-liter jug and y is the amount in the 3-liter jug. The goal is to get exactly 2 liters in either jug."
    ),
    Problem(
        name="Missionaries and Cannibals",
        description="Solve the Missionaries and Cannibals problem. The state is represented as (m, c, b) where m is the number of missionaries on the left bank, c is the number of cannibals on the left bank, and b is 1 if the boat is on the left bank, 0 otherwise. Start with (3, 3, 1) and the goal is (0, 0, 0)."
    ),
    Problem(
        name="N-Queens",
        description="Solve the N-Queens problem for an 8x8 board. The state is represented as a list of 8 integers, where each integer represents the column position of the queen in that row. The goal is to place 8 queens on the board so that no two queens threaten each other.",
        use_dfs=True
    ),
    Problem(
        name="Game 24",
        description="Solve the Game 24 problem. Given four integers between 1 and 9, find a way to arrive at 24 using only addition, subtraction, multiplication, and division. The state is represented as a list of numbers and intermediate results. The goal is to reach 24 using all initial numbers.",
        use_dfs=True
    )
]

for problem in problems:
    print(f"\nSolving {problem.name}:")
    if problem.name == "8-Puzzle":
        initial_state = [3,1,2,4,0,5,6,7,8]
    elif problem.name == "Tower of Hanoi":
        initial_state = [[3,2,1], [], []]
    elif problem.name == "Water Jug":
        initial_state = (0, 0)
    elif problem.name == "Missionaries and Cannibals":
        initial_state = (3, 3, 1)
    elif problem.name == "N-Queens":
        initial_state = [0] * 8
    elif problem.name == "Game 24":
        initial_state = [4, 7, 8, 8]  # Example initial numbers
    
    solution = solve_problem(problem, initial_state)
    if solution:
        print(f"Solution found in {len(solution)} steps:")
        for step, state in enumerate(solution):
            print(f"Step {step}: {state}")
    else:
        print("No solution found within the maximum number of iterations.")
        print("Generated successor function:")
        print(generate_search_components(problem).successor_function)
        print("Generated goal test function:")
        print(generate_search_components(problem).goal_test)
        if problem.use_astar:
            print("Generated heuristic function:")
            print(generate_search_components(problem).heuristic_function)

"""
Core Concepts and Workflow:
1. LLM-Generated Search Components: The script uses GPT-4-Turbo to generate problem-specific 
   successor functions, goal tests, and heuristic functions.
2. Minimal LLM Calls: Only one LLM call is made per problem to generate all necessary components.
3. Flexible Search Algorithms: The script implements BFS, DFS, and A* search algorithms, 
   choosing the appropriate one based on the problem characteristics.
4. Problem Abstraction: Problems are defined using a Pydantic model, allowing for easy addition 
   of new problem types.

Workflow:
1. Define the problem using the Problem class.
2. Generate search components using the LLM.
3. Execute the generated functions in a controlled namespace.
4. Apply the appropriate search algorithm using the generated components.
5. Return and display the solution if found.

Future Improvements:
1. Implement verification mechanisms for the soundness and completeness of generated components.
2. Add interactive refinement of generated components based on human feedback.
3. Implement more sophisticated prompting techniques to guide the LLM in generating more 
   efficient components.
4. Add support for more diverse problem types and search algorithms.
5. Implement performance metrics to measure and optimize computational efficiency.
6. Add error handling and timeout mechanisms for long-running searches.
7. Implement a caching system to store and reuse previously generated components for similar problems.
8. Develop a user interface for easier problem definition and result visualization.
"""