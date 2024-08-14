import os
import dspy
import openai
from typing import List, Tuple
import random
import math

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up DSPy with GPT-4
dspy.settings.configure(lm=dspy.OpenAI(model="gpt-4o-mini"))

class OpenTableEnvironment:
    def __init__(self):
        self.state = "homepage"
        self.date = None
        self.time = None
        self.party_size = None
        self.restaurant = None
    
    def step(self, action: str) -> Tuple[str, float, bool]:
        print(f"  Environment: Executing action '{action}'")
        if action.startswith("SELECT_DATE"):
            self.date = action.split()[1]
            self.state = "date_selected"
        elif action.startswith("SELECT_TIME"):
            self.time = action.split()[1]
            self.state = "time_selected"
        elif action.startswith("SELECT_PARTY_SIZE"):
            self.party_size = action.split()[1]
            self.state = "party_size_selected"
        elif action.startswith("SEARCH_RESTAURANT"):
            self.restaurant = action.split()[1]
            self.state = "restaurant_page"
        elif action == "COMPLETE_RESERVATION":
            if self.date and self.time and self.party_size and self.restaurant:
                print("  Environment: Reservation completed successfully!")
                return "Reservation completed", 1.0, True
        
        print(f"  Environment: New state - {self.state}")
        return f"Current state: {self.state}", 0.0, False

class RAGDrafter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_action = dspy.ChainOfThought("state, query -> action, rationale")
    
    def forward(self, state: str, query: str) -> Tuple[str, str]:
        print(f"  RAGDrafter: Generating action for state '{state}'")
        result = self.generate_action(state=state, query=query)
        action = result.action
        rationale = result.rationale
        
        # Ensure the action is one of the allowed actions
        allowed_actions = ["SELECT_DATE", "SELECT_TIME", "SELECT_PARTY_SIZE", "SEARCH_RESTAURANT", "COMPLETE_RESERVATION"]
        if not any(action.startswith(allowed) for allowed in allowed_actions):
            action = random.choice(allowed_actions)
            if action != "COMPLETE_RESERVATION":
                action += " placeholder"
        
        print(f"  RAGDrafter: Generated action '{action}' with rationale '{rationale}'")
        return action, rationale

class RAGVerifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.verify = dspy.ChainOfThought("state, action, rationale -> numeric_score")
    
    def forward(self, state: str, action: str, rationale: str) -> float:
        print(f"  RAGVerifier: Verifying action '{action}' for state '{state}'")
        result = self.verify(state=state, action=action, rationale=rationale)
        try:
            score = float(result.numeric_score)
        except ValueError:
            print(f"  RAGVerifier: Warning - Could not convert score '{result.numeric_score}' to float. Using default score of 5.")
            score = 5.0
        print(f"  RAGVerifier: Assigned score {score}")
        return score

class MCTSNode:
    def __init__(self, state: str, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

def mcts_search(env: OpenTableEnvironment, root_state: str, drafter: RAGDrafter, verifier: RAGVerifier, query: str, num_simulations: int = 10, max_depth: int = 5):
    print(f"\nStarting MCTS Search with {num_simulations} simulations")
    root = MCTSNode(root_state)
    
    for i in range(num_simulations):
        print(f"\nMCTS Simulation {i+1}/{num_simulations}")
        node = root
        depth = 0
        
        # Selection and Expansion
        while node.children and depth < max_depth:
            print("  MCTS: Selection phase")
            node = max(node.children, key=lambda n: n.value / (n.visits + 1) + math.sqrt(2 * math.log(n.parent.visits + 1) / (n.visits + 1)))
            depth += 1
        
        if depth < max_depth:
            print("  MCTS: Expansion phase")
            action, rationale = drafter(node.state, query)
            child_state, reward, done = env.step(action)
            child = MCTSNode(child_state, parent=node)
            node.children.append(child)
            node = child
            depth += 1
        
        # Simulation
        print("  MCTS: Simulation phase")
        while not done and depth < max_depth:
            action, _ = drafter(node.state, query)
            child_state, reward, done = env.step(action)
            child = MCTSNode(child_state, parent=node)
            node.children.append(child)
            node = child
            depth += 1
        
        # Backpropagation
        print("  MCTS: Backpropagation phase")
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    best_child = max(root.children, key=lambda n: n.visits)
    print(f"\nMCTS Search completed. Best state: {best_child.state}")
    return best_child.state

def dpo_train(drafter: RAGDrafter, verifier: RAGVerifier, dataset: List[Tuple[str, str, str, str, float]]):
    print("\nStarting DPO Training")
    dpo_loss = 0
    for i, (state, query, action_w, action_l, reward) in enumerate(dataset):
        print(f"\nTraining on example {i+1}/{len(dataset)}")
        print(f"  State: {state}")
        print(f"  Query: {query}")
        print(f"  Preferred action: {action_w}")
        print(f"  Non-preferred action: {action_l}")
        
        _, rationale_w = drafter(state, query)
        _, rationale_l = drafter(state, query)
        
        score_w = verifier(state, action_w, rationale_w)
        score_l = verifier(state, action_l, rationale_l)
        
        example_loss = math.log(1 / (1 + math.exp(score_l - score_w)))
        dpo_loss += example_loss
        print(f"  Example loss: {example_loss}")
    
    print(f"\nDPO Training completed. Total loss: {dpo_loss}")

if __name__ == "__main__":
    env = OpenTableEnvironment()
    drafter = RAGDrafter()
    verifier = RAGVerifier()
    
    query = "Book a reservation for 2 people at The French Laundry on June 15, 2024 at 7:00 PM"
    print(f"\nExecuting query: {query}")
    
    # MCTS Search
    best_state = mcts_search(env, env.state, drafter, verifier, query)
    print(f"Best state after MCTS: {best_state}")
    
    # Generate some example data for DPO training
    dataset = [
        ("homepage", query, "SELECT_DATE 2024-06-15", "SELECT_DATE 2024-06-14", 1.0),
        ("homepage", query, "SEARCH_RESTAURANT The_French_Laundry", "SEARCH_RESTAURANT Random_Restaurant", 1.0),
        ("restaurant_page", query, "SELECT_PARTY_SIZE 2", "SELECT_PARTY_SIZE 4", 1.0),
        ("reservation_page", query, "COMPLETE_RESERVATION", "SELECT_TIME 6:00_PM", 1.0)
    ]
    
    # DPO Training
    dpo_train(drafter, verifier, dataset)