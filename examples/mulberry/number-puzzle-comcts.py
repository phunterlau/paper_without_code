import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import json
import math
import random
from enum import Enum
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class ReasoningNode:
    """Represents a node in the reasoning tree"""
    content: str
    score: float = 0.0
    visits: int = 0
    parent: Optional['ReasoningNode'] = None
    children: List['ReasoningNode'] = None
    model_source: str = ""
    grid: List[List[int]] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.grid is None:
            self.grid = [[0, 0], [0, 0]]  # Initialize empty 2x2 grid

@dataclass
class ModelConfig:
    """Configuration for a model in collective search"""
    name: str
    temperature: float
    description: str

class ModelType(Enum):
    """Available models for collective search, simulated with different temperatures"""
    CONSERVATIVE = ModelConfig(
        name="gpt-4o-mini",
        temperature=0.1,
        description="Conservative reasoning with high precision"
    )
    BALANCED = ModelConfig(
        name="gpt-4o-mini",
        temperature=0.5,
        description="Balanced exploration and exploitation"
    )
    CREATIVE = ModelConfig(
        name="gpt-4o-mini",
        temperature=1.0,
        description="Creative reasoning with more exploration"
    )

PUZZLE_DESCRIPTION = """
Place numbers 1-4 in a 2x2 grid to satisfy:
- Each row and column contains each number exactly once
- Number in (1,1) is even
- Number in (2,2) is greater than number in (1,2)
- Sum of first row is greater than sum of second row
"""

class CoMCTS:
    """Collective Monte Carlo Tree Search with Reflective Reasoning"""
    
    def __init__(self, models: List[ModelType], max_iterations: int = 50, exploration_constant: float = 1.4):
        self.models = models
        self.max_iterations = max_iterations
        self.exploration_constant = exploration_constant

    def verify_constraints(self, grid: List[List[int]]) -> List[str]:
        """Verify which constraints are violated by current grid"""
        violated = []

        if not all(x in [0, 1, 2, 3, 4] for row in grid for x in row):
            return ["Invalid numbers in grid"]

        # Check for complete grid
        if any(x == 0 for row in grid for x in row):
            return []  # Incomplete grid is not violation

        # Check row uniqueness
        for i, row in enumerate(grid):
            if len(set(row)) != len(row):
                violated.append(f"Row {i+1} contains duplicates")

        # Check column uniqueness
        for j in range(len(grid[0])):
            col = [grid[i][j] for i in range(len(grid))]
            if len(set(col)) != len(col):
                violated.append(f"Column {j+1} contains duplicates")

        # Check (1,1) is even
        if grid[0][0] % 2 != 0:
            violated.append("Number in (1,1) must be even")

        # Check (2,2) > (1,2)
        if grid[1][1] <= grid[0][1]:
            violated.append("Number in (2,2) must be greater than (1,2)")

        # Check first row sum > second row sum
        if sum(grid[0]) <= sum(grid[1]):
            violated.append("Sum of first row must be greater than second row")

        return violated

    def select_best_node(self, node: ReasoningNode) -> ReasoningNode:
        """Select best child node using UCB1 formula"""
        if not node.children:
            return node

        log_parent_visits = math.log(node.visits) if node.visits > 0 else 0
        
        return max(node.children, key=lambda c: float('inf') if c.visits == 0 else (
            c.score/c.visits + self.exploration_constant * math.sqrt(log_parent_visits/c.visits)
        ))

    def _print_grid(self, grid: List[List[int]]):
        """Helper function to print grid state"""
        print("\n┌───┬───┐")
        for i, row in enumerate(grid):
            print("│", end=" ")
            for cell in row:
                if cell == 0:
                    print("·", end=" │ ")
                else:
                    print(cell, end=" │ ")
            print()
            if i < len(grid) - 1:
                print("├───┼───┤")
        print("└───┴───┘")

    def expand_node(self, node: ReasoningNode) -> List[ReasoningNode]:
        """Expand current node using collective knowledge"""
        print("\n🌱 EXPANSION Phase")
        print("Current grid state:")
        self._print_grid(node.grid)
        new_nodes = []
        
        # Find first empty cell
        empty_cells = []
        for i in range(2):
            for j in range(2):
                if node.grid[i][j] == 0:
                    empty_cells.append((i, j))
        
        if not empty_cells:
            return []
            
        i, j = empty_cells[0]
        
        # Generate possible moves from each model
        for model in self.models:
            try:
                messages = [
                    {"role": "system", "content": "You are an expert in logical puzzles and number placement."},
                    {"role": "user", "content": f"""
Current grid state:
{node.grid}

Rules:
{PUZZLE_DESCRIPTION}

Suggest valid numbers (1-4) for position ({i+1},{j+1}).
Consider existing numbers and constraints.

Respond in JSON format:
{{
    "suggestions": [
        {{
            "number": 2,
            "reasoning": "This number is valid because...",
            "confidence": 0.9
        }}
    ]
}}"""}
                ]
                
                response = client.chat.completions.create(
                    model=model.value.name,
                    temperature=model.value.temperature,
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                print(f"\nUsing {model.name} (temp={model.value.temperature})")
                print(f"Purpose: {model.value.description}")
                
                result = json.loads(response.choices[0].message.content)
                
                for suggestion in result["suggestions"]:
                    print(f"\nModel {model.value} suggests {suggestion['number']} because:")
                    print(suggestion["reasoning"])
                    
                    new_grid = [row.copy() for row in node.grid]
                    new_grid[i][j] = suggestion["number"]
                    
                    # Check constraints before adding
                    violations = self.verify_constraints(new_grid)
                    if not violations:
                        print(f"✅ Valid move! Adding node for number {suggestion['number']}")
                        new_node = ReasoningNode(
                            content=suggestion["reasoning"],
                            parent=node,
                            model_source=model.value,
                            grid=new_grid
                        )
                        node.children.append(new_node)
                        new_nodes.append(new_node)
                    else:
                        print(f"❌ Invalid move! Violations:")
                        for violation in violations:
                            print(f"  - {violation}")
                
            except Exception as e:
                print(f"Error expanding with model {model}: {str(e)}")
                continue
                
        return new_nodes

    def simulate_and_evaluate(self, node: ReasoningNode) -> float:
        """Evaluate node quality using collective knowledge"""
        print("\n📊 SIMULATION & EVALUATION Phase")
        print("Evaluating grid:")
        self._print_grid(node.grid)
        
        if not node.grid:
            print("❌ Invalid grid state")
            return 0.0

        # Count filled cells (progress metric)
        filled = sum(1 for row in node.grid for cell in row if cell != 0)
        progress_score = filled / 4  # 4 cells total
        print(f"Progress score: {progress_score:.2f} ({filled}/4 cells filled)")

        # Check constraints
        violations = self.verify_constraints(node.grid)
        constraint_score = 1.0 if not violations else max(0, 1 - (len(violations) * 0.2))
        print(f"Constraint score: {constraint_score:.2f}")
        if violations:
            print("Violations found:")
            for violation in violations:
                print(f"  - {violation}")

        # Combine scores
        final_score = (progress_score + constraint_score) / 2
        print(f"Final evaluation score: {final_score:.2f}")
        return final_score

    def backpropagate(self, node: ReasoningNode, score: float):
        """Update node statistics back to root"""
        print("\n⬆️ BACKPROPAGATION Phase")
        current = node
        depth = 0
        while current:
            old_score = current.score / max(current.visits, 1)
            current.visits += 1
            current.score += score
            new_score = current.score / current.visits
            
            print(f"Level {depth}:")
            print(f"  Visits: {current.visits}")
            print(f"  Score: {old_score:.2f} → {new_score:.2f}")
            
            current = current.parent
            depth += 1
        print(f"Backpropagated through {depth} levels")

    def search(self) -> Dict[str, Any]:
        """Main CoMCTS search algorithm"""
        print("\n🔍 Starting CoMCTS Search")
        root = ReasoningNode(content="Start", grid=[[0, 0], [0, 0]])
        iterations = 0
        print("Initial grid:")
        self._print_grid(root.grid)
        
        while iterations < self.max_iterations:
            print(f"\n=========== Iteration {iterations + 1}/{self.max_iterations} ===========")
            
            # Selection
            print("\n🔍 SELECTION Phase")
            current = root
            selection_path = []
            while current.children and not self._is_terminal(current):
                current = self.select_best_node(current)
                selection_path.append(f"Grid state (score: {current.score/max(1, current.visits):.2f})")
            
            if selection_path:
                print("Selection path:")
                for step in selection_path:
                    print(f"- {step}")
            else:
                print("At root node")
            
            # Expansion
            if not self._is_terminal(current):
                new_nodes = self.expand_node(current)
                if new_nodes:
                    current = random.choice(new_nodes)
            
            # Simulation
            score = self.simulate_and_evaluate(current)
            
            # Backpropagation
            self.backpropagate(current, score)
            
            # Check for solution
            if score > 0.95 and self._is_terminal(current):
                return {
                    "success": True,
                    "grid": current.grid,
                    "iterations": iterations + 1,
                    "reasoning_path": self._get_path_to_node(current)
                }
            
            iterations += 1
        
        # Return best found solution
        best_node = self._get_best_node(root)
        return {
            "success": False,
            "grid": best_node.grid,
            "iterations": iterations,
            "reasoning_path": self._get_path_to_node(best_node)
        }

    def _is_terminal(self, node: ReasoningNode) -> bool:
        """Check if node represents a complete valid solution"""
        if not node.grid:
            return False
            
        # Check if grid is complete
        if any(cell == 0 for row in node.grid for cell in row):
            return False
            
        # Check if all constraints are satisfied
        return not self.verify_constraints(node.grid)

    def _get_path_to_node(self, node: ReasoningNode) -> List[str]:
        """Get reasoning path from root to current node"""
        path = []
        current = node
        while current:
            if current.content != "Start":
                path.append(current.content)
            current = current.parent
        return list(reversed(path))

    def _get_best_node(self, root: ReasoningNode) -> ReasoningNode:
        """Get node with best score in tree"""
        best_score = float('-inf')
        best_node = root
        
        def search(node):
            nonlocal best_score, best_node
            if node.visits > 0 and node.score/node.visits > best_score:
                best_score = node.score/node.visits
                best_node = node
            for child in node.children:
                search(child)
        
        search(root)
        return best_node

def run_number_puzzle():
    """Run the sequential number logic puzzle"""
    models = [
        ModelType.CONSERVATIVE,  # Temperature 0.1 for precise reasoning
        ModelType.BALANCED,      # Temperature 0.5 for balanced approach
        ModelType.CREATIVE       # Temperature 1.0 for exploratory reasoning
    ]
    
    print("\n🤖 Model Configuration:")
    for model in models:
        print(f"\n{model.name}:")
        print(f"- Temperature: {model.value.temperature}")
        print(f"- Role: {model.value.description}")
    
    print("Starting Sequential Number Logic Puzzle...")
    print("\nPuzzle:")
    print(PUZZLE_DESCRIPTION)
    
    comcts = CoMCTS(models)
    result = comcts.search()
    
    print("\nSearch Results:")
    print(f"Found solution: {result['success']}")
    print(f"Completed in {result['iterations']} iterations")
    
    print("\nFinal Grid:")
    for row in result['grid']:
        print(row)
    
    print("\nReasoning Path:")
    for i, step in enumerate(result['reasoning_path'], 1):
        print(f"\nStep {i}:")
        print(step)

if __name__ == "__main__":
    run_number_puzzle()
