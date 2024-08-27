import os
import random
import numpy as np
from typing import List, Tuple, Dict
import openai
from collections import deque
from openai import OpenAI

from graphviz import Digraph
import time 
import colorsys

# Set up OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

print("Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents")
print("================================================================")

class WebShop:
    def __init__(self):
        self.items = [
            {"id": 1, "name": "Laptop", "price": 999, "category": "Electronics"},
            {"id": 2, "name": "Smartphone", "price": 599, "category": "Electronics"},
            {"id": 3, "name": "Headphones", "price": 199, "category": "Electronics"},
            {"id": 4, "name": "T-shirt", "price": 29, "category": "Clothing"},
            {"id": 5, "name": "Jeans", "price": 59, "category": "Clothing"},
        ]
        self.current_page = "home"
        self.cart = []
        self.search_query = ""
        self.current_item = None  # Initialize current_item

    def get_observation(self) -> str:
        if self.current_page == "home":
            return """
            <html>
                <body>
                    <h1>WebShop Home</h1>
                    <input type="text" id="search_bar" placeholder="Search for items">
                    <button id="search_button">Search</button>
                    <button id="view_cart">View Cart</button>
                </body>
            </html>
            """
        elif self.current_page == "search_results":
            items_html = "".join([f'<div class="item" id="item_{item["id"]}">{item["name"]} - ${item["price"]}</div>' for item in self.items if self.search_query.lower() in item["name"].lower()])
            return f"""
            <html>
                <body>
                    <h1>Search Results for "{self.search_query}"</h1>
                    {items_html}
                    <button id="back_to_home">Back to Home</button>
                </body>
            </html>
            """
        elif self.current_page == "item_details":
            if self.current_item is not None:
                item = self.items[self.current_item]
                return f"""
                <html>
                    <body>
                        <h1>{item["name"]}</h1>
                        <p>Price: ${item["price"]}</p>
                        <p>Category: {item["category"]}</p>
                        <button id="add_to_cart">Add to Cart</button>
                        <button id="back_to_results">Back to Results</button>
                    </body>
                </html>
                """
            else:
                return "Error: No item selected"
        elif self.current_page == "cart":
            cart_items = "".join([f'<div class="cart_item">{item["name"]} - ${item["price"]}</div>' for item in self.cart])
            return f"""
            <html>
                <body>
                    <h1>Shopping Cart</h1>
                    {cart_items}
                    <button id="checkout">Checkout</button>
                    <button id="back_to_home">Back to Home</button>
                </body>
            </html>
            """

    def take_action(self, action: str) -> Tuple[str, float, bool]:
        print(f"Taking action: {action}")
        if action.startswith("search"):
            self.search_query = action.split(" ", 1)[1]
            self.current_page = "search_results"
            return self.get_observation(), 0, False
        elif action.startswith("view_item"):
            item_id = int(action.split()[-1])
            self.current_item = item_id - 1
            self.current_page = "item_details"
            return self.get_observation(), 0, False
        elif action == "add_to_cart":
            if self.current_item is not None:
                self.cart.append(self.items[self.current_item])
                return f"Added {self.items[self.current_item]['name']} to cart", 0.1, False
            else:
                return "Error: No item selected", -0.1, False
        elif action == "view_cart":
            self.current_page = "cart"
            return self.get_observation(), 0, False
        elif action == "checkout":
            if len(self.cart) > 0:
                return "Checkout successful", 1, True
            else:
                return "Cart is empty", -0.1, False
        elif action == "back_to_home":
            self.current_page = "home"
            self.current_item = None
            return self.get_observation(), 0, False
        elif action == "back_to_results":
            self.current_page = "search_results"
            self.current_item = None
            return self.get_observation(), 0, False
        else:
            return "Invalid action", -0.1, False

class DPO:
    def __init__(self, beta=1.0):
        self.beta = beta
        self.reference_model = None  # This should be initialized with the initial policy

    def optimize(self, preference_pairs):
        # Implement Direct Preference Optimization
        print("Optimizing policy using Direct Preference Optimization (DPO)")
        losses = []
        for h, a_w, a_l in preference_pairs:
            # In a full implementation, these would be computed using the actual policy and reference model
            pi_w, pi_l = np.random.random(), np.random.random()
            ref_w, ref_l = np.random.random(), np.random.random()
            
            loss = -np.log(1 / (1 + np.exp(self.beta * (np.log(pi_l / ref_l) - np.log(pi_w / ref_w)))))
            losses.append(loss)
        
        avg_loss = np.mean(losses)
        print(f"Average DPO loss: {avg_loss}")
        return avg_loss

class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        # Add experience to the replay buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Sample a batch of experiences from the replay buffer
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

class AgentQ:
    def __init__(self, env: WebShop, mcts_simulations: int = 2, mcts_max_depth: int = 2, max_steps_per_episode: int = 100):
        self.env = env
        self.mcts = MCTS(env, num_simulations=mcts_simulations, max_depth=mcts_max_depth)
        self.dpo = DPO()
        self.replay_buffer = ReplayBuffer()
        self.value_threshold = 0.1
        self.episode_trees = []  # Store MCTS trees for each step
        self.max_steps_per_episode = max_steps_per_episode

    def get_action(self, observation: str) -> str:
        print("Agent Q: Selecting action using MCTS")
        action, root = self.mcts.search(observation)
        self.episode_trees.append(root)  # Store the root of the MCTS tree
        return action

    def generate_preference_pairs(self, node):
        print("Generating preference pairs for DPO")
        pairs = []
        for i, child1 in enumerate(node.children):
            for j, child2 in enumerate(node.children[i+1:]):
                if abs(child1.value() - child2.value()) > self.value_threshold:
                    if child1.value() > child2.value():
                        pairs.append((node.observation, child1.action, child2.action))
                    else:
                        pairs.append((node.observation, child2.action, child1.action))
        print(f"Generated {len(pairs)} preference pairs")
        return pairs

    def train(self, num_episodes: int):
        print(f"Training Agent Q for {num_episodes} episodes")
        start_time = time.time()
        total_steps = 0
        total_rewards = []

        for episode in range(num_episodes):
            episode_start_time = time.time()
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            observation = self.env.get_observation()
            done = False
            total_reward = 0
            step = 0
            self.episode_trees = []  # Reset episode trees for new episode

            while not done and step < self.max_steps_per_episode:
                action = self.get_action(observation)
                next_observation, reward, done = self.env.take_action(action)
                self.replay_buffer.add(observation, action, reward, next_observation, done)
                total_reward += reward
                
                preference_pairs = self.generate_preference_pairs(self.episode_trees[-1])
                self.dpo.optimize(preference_pairs)
                
                observation = next_observation
                step += 1
                total_steps += 1

                # Print progress every 10 steps
                if step % 10 == 0:
                    print(f"  Step {step}: Current reward = {total_reward:.2f}")

            if step >= self.max_steps_per_episode:
                print(f"  Episode terminated after reaching max steps ({self.max_steps_per_episode})")

            episode_duration = time.time() - episode_start_time
            total_rewards.append(total_reward)
            avg_reward = sum(total_rewards) / len(total_rewards)
            
            # Visualize MCTS trees for this episode
            self.visualize_episode_mcts_trees(episode + 1, max_depth=3, max_children=5)

            print(f"Episode {episode + 1} completed:")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Steps taken: {step}")
            print(f"  Episode duration: {episode_duration:.2f} seconds")
            print(f"  Average reward so far: {avg_reward:.2f}")
            
            self.visualize_episode(episode + 1)  # Visualize the entire episode

        total_duration = time.time() - start_time
        print("\nTraining completed:")
        print(f"Total episodes: {num_episodes}")
        print(f"Total steps: {total_steps}")
        print(f"Average steps per episode: {total_steps / num_episodes:.2f}")
        print(f"Average reward: {sum(total_rewards) / len(total_rewards):.2f}")
        print(f"Total duration: {total_duration:.2f} seconds")
        print(f"Average duration per episode: {total_duration / num_episodes:.2f} seconds")

    def visualize_full_mcts_tree(self, episode_number, step_number, max_depth=None, max_children=None):
        dot = Digraph(comment=f'Episode {episode_number}, Step {step_number} Full MCTS Tree')
        dot.attr(rankdir='TB', size='30,30')

        def add_node(node, parent_id=None, depth=0):
            if max_depth is not None and depth > max_depth:
                return

            node_id = str(id(node))
            label = f"{node.action if node.action else 'root'}\n"
            label += f"Visits: {node.visits}\n"
            label += f"Value: {node.value():.2f}"

            color = self.get_color_for_value(node.value())
            shape = self.get_node_shape(node.action) if node.action else 'doubleoctagon'

            dot.node(node_id, label, style='filled', color=color, shape=shape)

            if parent_id:
                dot.edge(parent_id, node_id)

            children = node.children
            if max_children is not None and len(children) > max_children:
                # If we have too many children, select a subset
                children = sorted(children, key=lambda c: c.visits, reverse=True)[:max_children]
                children.append(MCTSNode("...", parent=node, action="(more)"))

            for child in children:
                add_node(child, node_id, depth + 1)

        root = self.episode_trees[step_number]
        add_node(root)

        # Add a legend
        with dot.subgraph(name='cluster_legend') as c:
            c.attr(label='Legend')
            c.node('legend_high', 'High Value', style='filled', color=self.get_color_for_value(1.0))
            c.node('legend_med', 'Medium Value', style='filled', color=self.get_color_for_value(0.5))
            c.node('legend_low', 'Low Value', style='filled', color=self.get_color_for_value(0.0))
            c.node('legend_search', 'Search Action', shape='diamond')
            c.node('legend_view', 'View Action', shape='ellipse')
            c.node('legend_cart', 'Cart Action', shape='box')

        filename = f'episode_{episode_number}_step_{step_number}_full_mcts_tree'
        dot.render(filename, view=True, format='png', cleanup=True)
        print(f"Full MCTS tree visualization saved as {filename}.png")

    def visualize_episode_mcts_trees(self, episode_number, max_depth=None, max_children=None):
        for step, tree in enumerate(self.episode_trees):
            self.visualize_full_mcts_tree(episode_number, step, max_depth, max_children)
    
    def visualize_episode(self, episode_number):
        dot = Digraph(comment=f'Episode {episode_number} MCTS Trees')
        dot.attr(rankdir='LR', size='30,30')

        # Add a legend
        with dot.subgraph(name='cluster_legend') as c:
            c.attr(label='Legend')
            c.node('legend_high', 'High Value', style='filled', color=self.get_color_for_value(1.0))
            c.node('legend_med', 'Medium Value', style='filled', color=self.get_color_for_value(0.5))
            c.node('legend_low', 'Low Value', style='filled', color=self.get_color_for_value(0.0))
            c.node('legend_search', 'Search Action', shape='diamond')
            c.node('legend_view', 'View Action', shape='ellipse')
            c.node('legend_cart', 'Cart Action', shape='box')

        for step, root in enumerate(self.episode_trees):
            with dot.subgraph(name=f'cluster_{step}') as c:
                c.attr(label=f'Step {step + 1}')
                self.add_nodes_edges(root, c, f's{step}_')

        dot.render(f'episode_{episode_number}_mcts_trees', view=True, format='png', cleanup=True)

    def add_nodes_edges(self, node, graph, prefix, parent_id=None, depth=0):
        if depth > self.mcts.max_depth:
            return

        node_id = f"{prefix}{id(node)}"
        label = f"{node.action if node.action else 'root'}\n"
        label += f"Visits: {node.visits}\n"
        label += f"Value: {node.value():.2f}"
        
        color = self.get_color_for_value(node.value())
        shape = self.get_node_shape(node.action) if node.action else 'doubleoctagon'
        
        graph.node(node_id, label, style='filled', color=color, shape=shape)
        
        if parent_id:
            graph.edge(parent_id, node_id)
        
        for child in node.children:
            self.add_nodes_edges(child, graph, prefix, node_id, depth + 1)

    def get_color_for_value(self, value):
        # Use HSV color space for a smooth transition from red to yellow to green
        hue = value * 0.3  # This will give a range from red (0) to green (0.3)
        saturation = 0.7  # Reduce saturation for less intense colors
        value = 0.9  # Keep brightness high but not maximum
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    def get_node_shape(self, action):
        if action.startswith('search'):
            return 'diamond'
        elif action.startswith('view'):
            return 'ellipse'
        elif action in ['add_to_cart', 'checkout']:
            return 'box'
        else:
            return 'oval'
    
    def self_critique(self, observation: str, action: str) -> float:
        print("Performing self-critique")
        prompt = f"""
        Given the current observation: "{observation}"
        And the proposed action: "{action}"
        
        Rate the quality of this action on a scale from 0 to 1, where 1 is the best possible action and 0 is the worst.
        Provide a brief explanation for your rating.
        
        Rating:
        Explanation:
        """
        
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI assistant evaluating the quality of actions in a web shopping environment."},
                {"role": "user", "content": prompt}
            ]
        )
        
        critique = response.choices[0].message.content
        rating = float(critique.split("Rating:")[1].split("\n")[0].strip())
        explanation = critique.split("Explanation:")[1].strip()
        print(f"Self-critique rating: {rating}")
        print(f"Explanation: {explanation}")
        return rating

class MCTSNode:
    def __init__(self, observation: str, parent: 'MCTSNode' = None, action: str = None):
        self.observation = observation
        self.parent = parent
        self.action = action
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.total_value = 0
        self.untried_actions = self.get_possible_actions()

    def get_possible_actions(self):
        # Define possible actions for the node
        return ["search laptop", "search smartphone", "view_item 1", "view_item 2", "add_to_cart", "view_cart", "checkout", "back_to_home"]

    def fully_expanded(self):
        # Check if all possible actions have been tried
        return len(self.untried_actions) == 0

    def get_untried_action(self):
        # Get an untried action
        if not self.untried_actions:
            return None
        return self.untried_actions.pop()

    def add_child(self, child_node):
        # Add a child node
        self.children.append(child_node)

    def update(self, value):
        # Update the node's statistics
        self.visits += 1
        self.total_value += value

    def value(self):
        # Calculate the node's value
        return self.total_value / self.visits if self.visits > 0 else 0

    def ucb_score(self, c=1.41):
        # Calculate the UCB1 score for the node
        if self.visits == 0:
            return float('inf')
        return self.value() + c * np.sqrt(np.log(self.parent.visits) / self.visits)

class MCTS:
    def __init__(self, env: WebShop, num_simulations: int = 5, max_depth: int = 3):
        self.env = env
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.root = None

    def search(self, observation: str) -> Tuple[str, MCTSNode]:
        print(f"Starting MCTS search with {self.num_simulations} simulations and max depth {self.max_depth}")
        self.root = MCTSNode(observation)
        
        for i in range(self.num_simulations):
            print(f"Simulation {i + 1}/{self.num_simulations}")
            node = self.select(self.root)
            value = self.simulate(node)
            self.backpropagate(node, value)
        
        best_child = self.best_child(self.root)
        if best_child:
            print(f"MCTS search completed. Best action: {best_child.action}")
            return best_child.action, self.root
        else:
            print("MCTS search failed to find a best action. Returning a random action.")
            return random.choice(self.root.get_possible_actions()), self.root

    def select(self, node, depth=0):
        print(f"MCTS: Selection phase (depth {depth})")
        while node.children and depth < self.max_depth:
            if not node.fully_expanded():
                return self.expand(node, depth)
            else:
                node = self.ucb_select(node)
            depth += 1
        if depth < self.max_depth and not node.fully_expanded():
            return self.expand(node, depth)
        return node

    def expand(self, node, depth):
        print(f"MCTS: Expansion phase (depth {depth})")
        action = node.get_untried_action()
        if action is None:
            # If no untried actions, select a random action
            action = random.choice(node.get_possible_actions())
        next_observation, reward, done = self.env.take_action(action)
        child = MCTSNode(next_observation, parent=node, action=action)
        node.add_child(child)
        return child

    def simulate(self, node):
        print(f"MCTS: Simulation phase (depth {self.get_node_depth(node)})")
        return self.ai_process_supervision(node.observation, node.action)

    def backpropagate(self, node, value):
        print(f"MCTS: Backpropagation phase (starting from depth {self.get_node_depth(node)})")
        while node is not None:
            node.update(value)
            node = node.parent

    def get_node_depth(self, node):
        depth = 0
        while node.parent is not None:
            depth += 1
            node = node.parent
        return depth

    def best_child(self, node):
        if not node.children:
            return None
        return max(node.children, key=lambda c: c.visits)

    def ucb_select(self, node):
        return max(node.children, key=lambda c: c.ucb_score())

    def ai_process_supervision(self, observation: str, action: str) -> float:
        #print("Performing AI process supervision using GPT-4")
        prompt = f"""
        You are an AI assistant evaluating the quality of actions in a web shopping environment.
        
        Current webpage observation:
        {observation}
        
        Proposed action:
        {action}
        
        Rate the quality of this action on a scale from 0 to 1, where 1 is the best possible action and 0 is the worst.
        Consider factors such as relevance to the current page, progression towards a shopping goal, and adherence to typical web navigation patterns.
        
        Provide your rating and a brief explanation in the following format:
        
        Rating: [Your rating between 0 and 1]
        Explanation: [Your brief explanation]
        """
        
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI assistant evaluating web navigation actions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150
            )
            
            critique = response.choices[0].message.content
            rating_line = [line for line in critique.split('\n') if line.startswith("Rating:")][0]
            rating = float(rating_line.split(":")[1].strip())
            
            print(f"AI Process Supervision - Action: {action}, Rating: {rating}")
            print(f"Explanation: {critique.split('Explanation:')[1].strip()}")
            
            return rating
        except Exception as e:
            print(f"Error in AI process supervision: {e}")
            return random.random()  # Fallback to random rating in case of API error
    
    def get_color_for_value(self, value):
        # This function returns a color based on the node's value
        # Green for high values, red for low values
        r = int(255 * (1 - value))
        g = int(255 * value)
        b = 0
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def visualize_tree(self):
        dot = Digraph(comment='MCTS Tree')
        dot.attr(rankdir='TB', size='8,8')
        
        def add_nodes_edges(node, parent_id=None, depth=0):
            if depth > self.max_depth:
                return

            node_id = str(id(node))
            label = f"{node.action if node.action else 'root'}\n"
            label += f"Visits: {node.visits}\n"
            label += f"Value: {node.value():.2f}"
            
            color = self.get_color_for_value(node.value())
            
            dot.node(node_id, label, style='filled', color=color)
            
            if parent_id:
                dot.edge(parent_id, node_id)
            
            for child in node.children:
                add_nodes_edges(child, node_id, depth + 1)
        
        add_nodes_edges(self.root)
        dot.render('mcts_tree', view=True, format='png', cleanup=True)

class MCTSNode:
    def __init__(self, observation: str, parent: 'MCTSNode' = None, action: str = None):
        self.observation = observation
        self.parent = parent
        self.action = action
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.total_value = 0
        self.untried_actions = self.get_possible_actions()

    def get_possible_actions(self):
        # Define possible actions for the node
        return ["search laptop", "search smartphone", "view_item 1", "view_item 2", "add_to_cart", "view_cart", "checkout", "back_to_home"]

    def fully_expanded(self):
        # Check if all possible actions have been tried
        return len(self.untried_actions) == 0

    def get_untried_action(self):
        # Get an untried action
        if not self.untried_actions:
            return None
        return self.untried_actions.pop()

    def add_child(self, child_node):
        # Add a child node
        self.children.append(child_node)

    def update(self, value):
        # Update the node's statistics
        self.visits += 1
        self.total_value += value

    def value(self):
        # Calculate the node's value
        return self.total_value / self.visits if self.visits > 0 else 0

    def ucb_score(self, c=1.41):
        # Calculate the UCB1 score for the node
        if self.visits == 0:
            return float('inf')
        return self.value() + c * np.sqrt(np.log(self.parent.visits) / self.visits)

def main():
    print("Initializing WebShop environment and Agent Q")
    env = WebShop()
    agent = AgentQ(env)

    print("Training Agent Q")
    agent.train(num_episodes=1)  # You can adjust the number of episodes as needed

    print("\nTesting trained Agent Q")
    observation = env.get_observation()
    done = False
    total_reward = 0
    step = 0

    while not done:
        print(f"\nStep {step + 1}")
        print("Current Observation:")
        print(observation)
        
        action = agent.get_action(observation)
        print(f"\nChosen action: {action}")
        
        critique_score = agent.self_critique(observation, action)
        print(f"Self-critique score: {critique_score}")
        
        observation, reward, done = env.take_action(action)
        total_reward += reward
        print(f"Reward: {reward}")
        print(f"Total reward so far: {total_reward}")
        
        step += 1

    print(f"\nTest completed. Final total reward: {total_reward}")

if __name__ == "__main__":
    main()