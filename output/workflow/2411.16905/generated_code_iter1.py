
# Generated code based on the paper
# Template: pytorch_model.py
# Subfield: artificial_intelligence
# Iteration: 1/3

"""
Template for model-related papers using PyTorch.
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class SimpleModel(nn.Module):
    """
    A simple PyTorch model template.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the model.
        
        Args:
            input_size (int): The size of the input
            hidden_size (int): The size of the hidden layer
            output_size (int): The size of the output
        """
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): The input tensor
            
        Returns:
            torch.Tensor: The output tensor
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleDataset(Dataset):
    """
    A simple PyTorch dataset template.
    """
    def __init__(self, data, targets):
        """
        Initialize the dataset.
        
        Args:
            data (np.ndarray): The input data
            targets (np.ndarray): The target data
        """
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: The length of the dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Args:
            idx (int): The index of the item
            
        Returns:
            tuple: The data and target
        """
        return self.data[idx], self.targets[idx]

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    """
    Train the model.
    
    Args:
        model (nn.Module): The model to train
        dataloader (DataLoader): The dataloader for the training data
        criterion: The loss function
        optimizer: The optimizer
        num_epochs (int): The number of epochs to train for
        
    Returns:
        list: The training losses
    """
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for inputs, targets in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    return losses

def evaluate_model(model, dataloader, criterion):
    """
    Evaluate the model.
    
    Args:
        model (nn.Module): The model to evaluate
        dataloader (DataLoader): The dataloader for the evaluation data
        criterion: The loss function
        
    Returns:
        float: The evaluation loss
    """
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
    
    return running_loss / len(dataloader)

def generate_sample_data(num_samples=1000, input_size=10):
    """
    Generate sample data for demonstration.
    
    Args:
        num_samples (int): The number of samples to generate
        input_size (int): The size of each input
        
    Returns:
        tuple: The input data and target data
    """
    # Generate random input data
    X = np.random.randn(num_samples, input_size)
    
    # Generate target data (a simple function of the input)
    y = np.sum(X, axis=1, keepdims=True) * 0.1
    
    return X, y

def plot_losses(losses):
    """
    Plot the training losses.
    
    Args:
        losses (list): The training losses
    """
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

def main():
    """
    Main function to demonstrate the implementation.
    """
    parser = argparse.ArgumentParser(description="Train a simple PyTorch model")
    parser.add_argument("--input_size", type=int, default=10, help="Size of the input")
    parser.add_argument("--hidden_size", type=int, default=50, help="Size of the hidden layer")
    parser.add_argument("--output_size", type=int, default=1, help="Size of the output")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    args = parser.parse_args()
    
    # Generate sample data
    X, y = generate_sample_data(num_samples=1000, input_size=args.input_size)
    
    # Split into train and test sets
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create datasets and dataloaders
    train_dataset = SimpleDataset(X_train, y_train)
    test_dataset = SimpleDataset(X_test, y_test)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Create the model
    model = SimpleModel(args.input_size, args.hidden_size, args.output_size)
    
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train the model
    losses = train_model(model, train_dataloader, criterion, optimizer, num_epochs=args.num_epochs)
    
    # Evaluate the model
    test_loss = evaluate_model(model, test_dataloader, criterion)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Plot the training losses
    plot_losses(losses)

if __name__ == "__main__":
    main()


# Implementation based on the paper

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class AgentModel(nn.Module):
    """
    Basic language model for agents in the Socratic learning system.
    This represents the core capability of each agent to process and generate text.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(AgentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embedding_dim)
        
        if hidden is None:
            output, hidden = self.lstm(embedded)
        else:
            output, hidden = self.lstm(embedded, hidden)
        # output shape: (batch_size, seq_len, hidden_dim)
        
        output = self.fc(output)
        # output shape: (batch_size, seq_len, vocab_size)
        return output, hidden
    
    def predict_next_token(self, sequence, temperature=1.0):
        """Generate the next token given a sequence."""
        with torch.no_grad():
            output, _ = self.forward(sequence)
            # Get the last token prediction
            logits = output[:, -1, :] / temperature
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            # Sample from the distribution
            next_token = torch.multinomial(probs, 1)
        return next_token

class SocraticAgent:
    """
    Agent that can participate in language games and improve through feedback.
    This represents an individual participant in Socratic learning.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, learning_rate=0.001, name="Agent"):
        self.name = name
        self.model = AgentModel(vocab_size, embedding_dim, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.experiences = []  # Memory of past interactions
        self.performance_history = []  # Track agent's performance over time
        
    def generate_response(self, prompt, max_length=20, temperature=1.0):
        """Generate a response to a given prompt."""
        self.model.eval()
        
        # Convert prompt to tensor
        if isinstance(prompt, list):
            sequence = torch.tensor(prompt).unsqueeze(0)
        else:
            sequence = prompt.unsqueeze(0)
            
        # Initial response is just the prompt
        response = sequence.clone()
        
        # Generate new tokens
        for _ in range(max_length):
            next_token = self.model.predict_next_token(response, temperature)
            response = torch.cat([response, next_token], dim=1)
            
        return response.squeeze(0)[len(prompt):]  # Return only the new tokens
    
    def learn_from_experience(self, batch_size=32, epochs=5):
        """
        Update model based on stored experiences.
        This implements the learning loop for self-improvement.
        """
        if len(self.experiences) < batch_size:
            return  # Not enough data to learn from
        
        self.model.train()
        
        # Sample experiences
        random.shuffle(self.experiences)
        batches = [self.experiences[i:i+batch_size] for i in range(0, len(self.experiences), batch_size)]
        
        total_loss = 0
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in batches:
                inputs = []
                targets = []
                
                for seq, target, _ in batch:
                    inputs.append(seq[:-1])  # Input is all but last token
                    targets.append(target)  # Target is the expected output

                # Pad sequences to same length
                max_len = max(len(seq) for seq in inputs)
                padded_inputs = [seq + [0] * (max_len - len(seq)) for seq in inputs]
                padded_targets = [seq + [0] * (max_len - len(seq)) for seq in targets]
                
                # Convert to tensors
                inputs_tensor = torch.tensor(padded_inputs)
                targets_tensor = torch.tensor(padded_targets)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs, _ = self.model(inputs_tensor)
                
                # Reshape for loss calculation
                outputs = outputs.reshape(-1, outputs.shape[-1])
                targets_tensor = targets_tensor.reshape(-1)
                
                # Calculate loss
                loss = self.criterion(outputs, targets_tensor)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                
            total_loss += epoch_loss / len(batches)
            
        avg_loss = total_loss / epochs
        self.performance_history.append(avg_loss)
        return avg_loss

class LanguageGame:
    """
    Framework for conducting language games between agents.
    This implements the core concept of language games for Socratic learning.
    """
    def __init__(self, agents, vocab_size, max_turns=5):
        self.agents = agents
        self.vocab_size = vocab_size
        self.max_turns = max_turns
        self.game_history = []
        
    def run_debate_game(self, topic_tokens, judge=None):
        """
        Run a debate game where agents take turns elaborating on a topic.
        The agent that provides the most coherent contribution wins.
        """
        current_sequence = topic_tokens.copy()
        turns_history = []
        
        # Each agent takes a turn adding to the conversation
        for turn in range(self.max_turns):
            for agent in self.agents:
                response = agent.generate_response(current_sequence, max_length=10)
                response_tokens = response.tolist()
                
                # Add to the current sequence
                current_sequence = current_sequence + response_tokens
                
                turns_history.append({
                    "agent": agent.name,
                    "response": response_tokens
                })
        
        # Evaluate quality - in a real system, this could be a more sophisticated model
        if judge is None:
            # Simple heuristic: responses with more varied tokens are considered better
            scores = {}
            for agent in self.agents:
                agent_responses = [t["response"] for t in turns_history if t["agent"] == agent.name]
                flat_responses = [item for sublist in agent_responses for item in sublist]
                unique_tokens = len(set(flat_responses))
                total_tokens = len(flat_responses) if len(flat_responses) > 0 else 1
                scores[agent.name] = unique_tokens / total_tokens
        else:
            # Use judge model to evaluate
            scores = judge.evaluate_debate(current_sequence, turns_history)
        
        # Determine winner
        winner = max(scores.items(), key=lambda x: x[1])[0]
        
        # Provide feedback to agents
        for agent in self.agents:
            quality_score = scores[agent.name]
            feedback = 1 if agent.name == winner else 0
            
            # Store experience with feedback
            agent.experiences.append((topic_tokens, current_sequence, feedback))
            
        # Record game outcome
        self.game_history.append({
            "topic": topic_tokens,
            "turns": turns_history,
            "scores": scores,
            "winner": winner
        })
        
        return winner, scores
    
    def run_question_answering_game(self, question_tokens):
        """
        Run a Q&A game where agents answer a question and get feedback based 
        on the accuracy of their response.
        """
        answers = {}
        
        # Each agent provides an answer
        for agent in self.agents:
            response = agent.generate_response(question_tokens, max_length=15)
            response_tokens = response.tolist()
            answers[agent.name] = response_tokens
        
        # Evaluate answers - in a real system, this would involve a reference model
        # Here we use a simple heuristic
        scores = {}
        for agent_name, answer in answers.items():
            # Simple heuristic: longer answers with more unique tokens are considered better
            unique_tokens = len(set(answer))
            total_tokens = len(answer) if len(answer) > 0 else 1
            scores[agent_name] = unique_tokens / total_tokens
        
        # Determine best answer
        best_answerer = max(scores.items(), key=lambda x: x[1])[0]
        
        # Provide feedback to agents
        for agent in self.agents:
            quality_score = scores[agent.name]
            feedback = 1 if agent.name == best_answerer else 0
            
            # Store experience with feedback
            agent.experiences.append((question_tokens, answers[agent.name], feedback))
        
        # Record game outcome
        self.game_history.append({
            "question": question_tokens,
            "answers": answers,
            "scores": scores,
            "best_answer": best_answerer
        })
        
        return best_answerer, scores
    
    def run_games(self, num_games, game_type="debate"):
        """Run multiple games of the specified type."""
        results = {"wins": defaultdict(int), "scores": defaultdict(list)}
        
        for _ in tqdm(range(num_games), desc=f"Running {game_type} games"):
            # Generate random topic/question
            if game_type == "debate":
                topic = [random.randint(1, self.vocab_size-1) for _ in range(5)]  # Random topic
                winner, scores = self.run_debate_game(topic)
            else:  # question_answering
                question = [random.randint(1, self.vocab_size-1) for _ in range(7)]  # Random question
                winner, scores = self.run_question_answering_game(question)
            
            # Record results
            results["wins"][winner] += 1
            for agent_name, score in scores.items():
                results["scores"][agent_name].append(score)
                
            # Agents learn from experiences
            for agent in self.agents:
                agent.learn_from_experience()
                
        return results

class SocraticLearningFramework:
    """
    Framework for implementing Socratic learning through language games.
    This class orchestrates the entire learning system described in the paper.
    """
    def __init__(self, num_agents=3, vocab_size=1000, embedding_dim=64, hidden_dim=128):
        # Initialize agents
        self.agents = [
            SocraticAgent(vocab_size, embedding_dim, hidden_dim, name=f"Agent_{i}")
            for i in range(num_agents)
        ]
        
        # Initialize language game environment
        self.language_game = LanguageGame(self.agents, vocab_size)
        
        # Environment parameters
        self.vocab_size = vocab_size
        self.cycles_completed = 0
    
    def run_cycle(self, num_games=10, game_types=None):
        """
        Run a complete cycle of Socratic learning.
        A cycle consists of multiple language games followed by agent learning.
        """
        if game_types is None:
            game_types = ["debate", "question_answering"]
            
        cycle_results = {}
        
        # For each game type
        for game_type in game_types:
            results = self.language_game.run_games(num_games, game_type)
            cycle_results[game_type] = results
            
        # Track performance for all agents
        agent_performances = {}
        for agent in self.agents:
            if agent.performance_history:
                agent_performances[agent.name] = agent.performance_history[-1]
            
        self.cycles_completed += 1
        
        return {
            "cycle": self.cycles_completed,
            "results": cycle_results,
            "agent_performances": agent_performances
        }
    
    def run_recursive_improvement(self, num_cycles=5, games_per_cycle=20):
        """
        Run multiple cycles to demonstrate recursive self-improvement.
        This implements the paper's central concept of systems improving themselves.
        """
        all_results = []
        
        for _ in tqdm(range(num_cycles), desc="Running learning cycles"):
            cycle_result = self.run_cycle(num_games=games_per_cycle)
            all_results.append(cycle_result)
            
        return all_results
    
    def analyze_results(self, results):
        """Analyze the results of recursive improvement."""
        # Track wins over time for each agent
        wins_over_time = defaultdict(list)
        loss_over_time = defaultdict(list)
        
        for cycle_result in results:
            cycle_num = cycle_result["cycle"]
            
            # Aggregate wins across all game types
            cycle_wins = Counter()
            for game_type, game_results in cycle_result["results"].items():
                for agent_name, win_count in game_results["wins"].items():
                    cycle_wins[agent_name] += win_count
            
            # Record wins for each agent
            for agent in self.agents:
                wins_over_time[agent.name].append(cycle_wins[agent.name])
                if agent.performance_history:
                    loss_over_time[agent.name].append(agent.performance_history[-1])
        
        # Plot win trends
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        for agent_name, wins in wins_over_time.items():
            plt.plot(range(1, len(wins)+1), wins, label=agent_name)
        plt.xlabel('Learning Cycle')
        plt.ylabel('Number of Wins')
        plt.title('Agent Performance Over Time')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        for agent_name, losses in loss_over_time.items():
            plt.plot(range(1, len(losses)+1), losses, label=agent_name)
        plt.xlabel('Learning Cycle')
        plt.ylabel('Loss')
        plt.title('Agent Learning Progress')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('socratic_learning_results.png')
        plt.close()
        
        return {
            "wins_over_time": dict(wins_over_time),
            "loss_over_time": dict(loss_over_time)
        }

def main():
    """Demonstrate the Socratic Learning Framework with a simple example."""
    print("Initializing Socratic Learning Framework...")
    
    # Create framework with a small vocab size for demonstration
    framework = SocraticLearningFramework(
        num_agents=3,
        vocab_size=100,
        embedding_dim=32,
        hidden_dim=64
    )
    
    print(f"Created {len(framework.agents)} agents")
    
    # Run recursive improvement process
    print("Starting recursive self-improvement process...")
    results = framework.run_recursive_improvement(num_cycles=5, games_per_cycle=10)
    
    # Analyze and visualize results
    print("Analyzing results...")
    analysis = framework.analyze_results(results)
    
    # Show final performance
    print("\nFinal Performance Summary:")
    for agent_name, wins in analysis["wins_over_time"].items():
        print(f"{agent_name}: Total Wins = {sum(wins)}, Learning Curve = {analysis['loss_over_time'].get(agent_name, [])}")
    
    print("\nSocratic learning demonstration completed.")
    print("Results visualization saved to 'socratic_learning_results.png'")

if __name__ == "__main__":
    main()

