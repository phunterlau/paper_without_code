
# Generated code based on the paper
# Template: pytorch_model.py
# Subfield: artificial_intelligence
# Iteration: 2/3

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
import copy
import time
import os

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class AgentModel(nn.Module):
    """
    Basic language model for agents in the Socratic learning system.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(AgentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
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
    
    def get_config(self):
        """Return model configuration for self-modification."""
        return {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim
        }

class SocraticAgent:
    """
    Agent that can participate in language games and improve through feedback.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, learning_rate=0.001, name="Agent"):
        self.name = name
        self.model = AgentModel(vocab_size, embedding_dim, hidden_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        self.experiences = []  # Memory of past interactions
        self.performance_history = []  # Track agent's performance over time
        self.model_history = []  # Track model versions
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        
    def generate_response(self, prompt, max_length=20, temperature=1.0):
        """Generate a response to a given prompt."""
        self.model.eval()
        
        # Convert prompt to tensor if it's a list
        if isinstance(prompt, list):
            sequence = torch.tensor([prompt]).long()
        else:
            # Ensure it's a 2D tensor
            if prompt.dim() == 1:
                sequence = prompt.unsqueeze(0)
            else:
                sequence = prompt
                
        # Initial response is just the prompt
        response = sequence.clone()
        
        # Generate new tokens
        for _ in range(max_length):
            next_token = self.model.predict_next_token(response, temperature)
            response = torch.cat([response, next_token], dim=1)
            
        return response.squeeze(0)[len(prompt):]  # Return only the new tokens
    
    def learn_from_experience(self, batch_size=16, epochs=3):
        """Update model based on stored experiences."""
        if len(self.experiences) < batch_size:
            return 0  # Not enough data to learn from
        
        self.model.train()
        
        # Sample experiences
        random.shuffle(self.experiences)
        batches = [self.experiences[i:i+batch_size] for i in range(0, len(self.experiences), batch_size)]
        
        total_loss = 0
        batch_count = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch in batches:
                # Process and pad sequences
                inputs, targets, weights = [], [], []
                
                for input_seq, target_seq, feedback in batch:
                    # Convert to list if tensor
                    if isinstance(input_seq, torch.Tensor):
                        input_seq = input_seq.tolist()
                    if isinstance(target_seq, torch.Tensor):
                        target_seq = target_seq.tolist()
                        
                    # Handle case where sequences are nested lists
                    if isinstance(input_seq[0], list):
                        input_seq = input_seq[0]
                    if isinstance(target_seq[0], list):
                        target_seq = target_seq[0]
                        
                    inputs.append(input_seq)
                    targets.append(target_seq)
                    weights.append(feedback)
                
                # Find max length for this batch
                max_in_len = max(len(seq) for seq in inputs)
                max_target_len = max(len(seq) for seq in targets)
                
                # Pad sequences
                padded_inputs = [seq + [0] * (max_in_len - len(seq)) for seq in inputs]
                padded_targets = [seq + [0] * (max_target_len - len(seq)) for seq in targets]
                
                # Convert to tensors
                inputs_tensor = torch.tensor(padded_inputs).long()
                targets_tensor = torch.tensor(padded_targets).long()
                weights_tensor = torch.tensor(weights).float()
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs, _ = self.model(inputs_tensor)
                
                # Calculate loss - apply only on valid tokens
                loss = 0
                for i, (output, target, weight) in enumerate(zip(outputs, targets_tensor, weights_tensor)):
                    # Only consider up to the length of the target
                    target_len = len(targets[i])
                    valid_output = output[:target_len, :]
                    valid_target = target[:target_len]
                    
                    # Reshape for loss calculation
                    valid_output = valid_output.view(-1, valid_output.size(-1))
                    valid_target = valid_target.view(-1)
                    
                    # Weight the loss by feedback
                    sample_loss = self.criterion(valid_output, valid_target) * weight
                    loss += sample_loss
                
                # Average loss across batch
                batch_loss = loss / len(inputs)
                
                # Backward pass
                batch_loss.backward()
                self.optimizer.step()
                
                epoch_loss += batch_loss.item()
                batch_count += 1
                
            total_loss += epoch_loss / len(batches)
            
        avg_loss = total_loss / epochs
        self.performance_history.append(avg_loss)
        return avg_loss
    
    def save_model_checkpoint(self):
        """Save the current model state as a checkpoint."""
        checkpoint = {
            'model_state': copy.deepcopy(self.model.state_dict()),
            'config': self.model.get_config(),
            'performance': self.performance_history[-1] if self.performance_history else float('inf')
        }
        self.model_history.append(checkpoint)
        
        # Keep only the best 3 models to save memory
        if len(self.model_history) > 3:
            self.model_history = sorted(self.model_history, key=lambda x: x['performance'])[:3]
    
    def attempt_self_modification(self, modification_type='scaling'):
        """
        Attempt to modify the model architecture based on performance.
        This implements the self-referential capabilities described in the paper.
        """
        if len(self.performance_history) < 2:
            return False, "Not enough performance history"
        
        # Only attempt modification if performance is improving
        if self.performance_history[-1] > self.performance_history[-2]:
            return False, "Performance is not improving"
        
        current_config = self.model.get_config()
        
        # Save current model before modification
        self.save_model_checkpoint()
        
        if modification_type == 'scaling':
            # Scale up hidden dimension
            new_hidden_dim = int(current_config['hidden_dim'] * 1.2)
            
            # Create new model with increased capacity
            new_model = AgentModel(
                current_config['vocab_size'],
                current_config['embedding_dim'],
                new_hidden_dim
            )
            
            # Transfer knowledge where possible (embedding and output layers)
            new_model.embedding.weight.data = self.model.embedding.weight.data.clone()
            
            # Initialize LSTM weights - we can't directly transfer due to dimension mismatch
            
            # For the output layer, we can reuse the weights
            new_model.fc.weight.data[:, :current_config['hidden_dim']] = self.model.fc.weight.data.clone()
            new_model.fc.bias.data = self.model.fc.bias.data.clone()
            
            # Replace model and optimizer
            self.model = new_model
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            return True, f"Scaled hidden dimension from {current_config['hidden_dim']} to {new_hidden_dim}"
            
        elif modification_type == 'revert':
            # Revert to best performing model if available
            if not self.model_history:
                return False, "No model history available"
                
            # Find best performing model
            best_model = min(self.model_history, key=lambda x: x['performance'])
            
            # Create new model with same config
            new_model = AgentModel(
                best_model['config']['vocab_size'],
                best_model['config']['embedding_dim'],
                best_model['config']['hidden_dim']
            )
            
            # Load state from checkpoint
            new_model.load_state_dict(best_model['model_state'])
            
            # Replace model and optimizer
            self.model = new_model
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            
            return True, f"Reverted to best model with performance {best_model['performance']:.6f}"
            
        return False, "No modification made"

class FeedbackModel(nn.Module):
    """
    Model that evaluates the quality of language outputs.
    This implements more sophisticated feedback mechanisms.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(FeedbackModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embedding_dim)
        
        output, _ = self.lstm(embedded)
        # output shape: (batch_size, seq_len, hidden_dim)
        
        # Self-attention to focus on important parts of the sequence
        attn_output, _ = self.attention(output, output, output)
        
        # Global average pooling
        pooled = torch.mean(attn_output, dim=1)
        
        # Final layers
        x = self.relu(self.fc1(pooled))
        x = self.sigmoid(self.fc2(x))
        return x

class LanguageGame:
    """
    Framework for conducting language games between agents.
    """
    def __init__(self, agents, vocab_size, max_turns=5, use_feedback_model=False):
        self.agents = agents
        self.vocab_size = vocab_size
        self.max_turns = max_turns
        self.game_history = []
        
        # Initialize feedback model if requested
        self.feedback_model = None
        if use_feedback_model:
            self.feedback_model = FeedbackModel(vocab_size, 64, 128)
            # Pre-train feedback model with some initial data
            self._initialize_feedback_model()
    
    def _initialize_feedback_model(self):
        """Initialize the feedback model with some synthetic data."""
        # Generate synthetic data - in a real system this would use high-quality examples
        X = []
        y = []
        
        # Generate positive examples (coherent sequences)
        for _ in range(100):
            # Create a coherent sequence with repeating patterns
            length = random.randint(10, 30)
            seq = [random.randint(1, self.vocab_size-1) for _ in range(length)]
            X.append(seq)
            y.append(1.0)  # High quality
            
        # Generate negative examples (random sequences)
        for _ in range(100):
            # Create a random sequence
            length = random.randint(10, 30)
            seq = [random.randint(1, self.vocab_size-1) for _ in range(length)]
            X.append(seq)
            y.append(0.0)  # Low quality
            
        # Train the model
        optimizer = optim.Adam(self.feedback_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # Convert to tensors
        # Pad sequences
        max_len = max(len(seq) for seq in X)
        X_padded = [seq + [0] * (max_len - len(seq)) for seq in X]
        X_tensor = torch.tensor(X_padded).long()
        y_tensor = torch.tensor(y).float().view(-1, 1)
        
        # Train for a few epochs
        self.feedback_model.train()
        for _ in range(5):
            optimizer.zero_grad()
            outputs = self.feedback_model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
    
    def evaluate_response_quality(self, sequence):
        """Evaluate the quality of a response using the feedback model."""
        if self.feedback_model is None:
            # Fallback to simple heuristic
            unique_tokens = len(set(sequence))
            total_tokens = len(sequence) if len(sequence) > 0 else 1
            return unique_tokens / total_tokens
        
        # Convert sequence to tensor
        if isinstance(sequence, list):
            # Handle nested lists
            if sequence and isinstance(sequence[0], list):
                sequence = sequence[0]
            seq_tensor = torch.tensor([sequence]).long()
        else:
            seq_tensor = sequence.unsqueeze(0) if sequence.dim() == 1 else sequence
            
        # Evaluate with feedback model
        self.feedback_model.eval()
        with torch.no_grad():
            quality_score = self.feedback_model(seq_tensor).item()
        return quality_score
        
    def run_debate_game(self, topic_tokens):
        """
        Run a debate game where agents take turns elaborating on a topic.
        """
        # Ensure topic is a list, not a tensor
        if isinstance(topic_tokens, torch.Tensor):
            topic_tokens = topic_tokens.tolist()
            
        current_sequence = topic_tokens.copy()
        turns_history = []
        
        # Each agent takes a turn adding to the conversation
        for turn in range(self.max_turns):
            for agent in self.agents:
                # Convert current sequence to tensor for model input
                current_tensor = torch.tensor(current_sequence).long()
                
                # Generate response
                response = agent.generate_response(current_tensor, max_length=10)
                response_tokens = response.tolist()
                
                # Add to the current sequence
                current_sequence = current_sequence + response_tokens
                
                turns_history.append({
                    "agent": agent.name,
                    "response": response_tokens
                })
        
        # Evaluate quality for each agent
        scores = {}
        for agent in self.agents:
            agent_responses = [t["response"] for t in turns_history if t["agent"] == agent.name]
            
            # Evaluate each response individually and average the scores
            response_scores = []
            for response in agent_responses:
                quality_score = self.evaluate_response_quality(response)
                response_scores.append(quality_score)
                
            scores[agent.name] = sum(response_scores) / len(response_scores) if response_scores else 0
        
        # Determine winner
        winner = max(scores.items(), key=lambda x: x[1])[0]
        
        # Provide feedback to agents - feedback is normalized to range [0, 1]
        max_score = max(scores.values()) if scores else 1
        min_score = min(scores.values()) if scores else 0
        score_range = max(max_score - min_score, 0.001)  # Avoid division by zero
        
        for agent in self.agents:
            normalized_score = (scores[agent.name] - min_score) / score_range
            
            # Add noise to feedback to encourage exploration (simulating randomness in evaluation)
            feedback = normalized_score * (0.9 + 0.2 * random.random())
            feedback = min(max(feedback, 0.0), 1.0)  # Ensure it stays in [0, 1]
            
            # Store experience with feedback
            agent.experiences.append((topic_tokens, current_sequence, feedback))
            
        # Record game outcome
        self.game_history.append({
            "type": "debate",
            "topic": topic_tokens,
            "turns": turns_history,
            "scores": scores,
            "winner": winner
        })
        
        return winner, scores
    
    def run_question_answering_game(self, question_tokens):
        """
        Run a Q&A game where agents answer a question and get feedback.
        """
        # Ensure question is a list, not a tensor
        if isinstance(question_tokens, torch.Tensor):
            question_tokens = question_tokens.tolist()
            
        answers = {}
        answer_tokens = {}
        
        # Each agent provides an answer
        for agent in self.agents:
            # Convert question to tensor for model input
            question_tensor = torch.tensor(question_tokens).long()
            
            # Generate response
            response = agent.generate_response(question_tensor, max_length=15)
            response_tokens = response.tolist()
            answers[agent.name] = question_tokens + response_tokens
            answer_tokens[agent.name] = response_tokens
        
        # Evaluate answers
        scores = {}
        for agent_name, answer in answers.items():
            quality_score = self.evaluate_response_quality(answer)
            scores[agent_name] = quality_score
        
        # Determine best answer
        best_answerer = max(scores.items(), key=lambda x: x[1])[0]
        
        # Provide feedback to agents - normalize scores
        max_score = max(scores.values()) if scores else 1
        min_score = min(scores.values()) if scores else 0
        score_range = max(max_score - min_score, 0.001)  # Avoid division by zero
        
        for agent in self.agents:
            normalized_score = (scores[agent.name] - min_score) / score_range
            
            # Add noise to feedback
            feedback = normalized_score * (0.9 + 0.2 * random.random())
            feedback = min(max(feedback, 0.0), 1.0)  # Ensure it stays in [0, 1]
            
            # Store experience with feedback
            agent.experiences.append((question_tokens, answer_tokens[agent.name], feedback))
        
        # Record game outcome
        self.game_history.append({
            "type": "qa",
            "question": question_tokens,
            "answers": answers,
            "scores": scores,
            "best_answer": best_answerer
        })
        
        return best_answerer, scores
    
    def run_creative_writing_game(self, prompt_tokens):
        """
        Run a creative writing game where agents generate stories from a prompt.
        This game type increases the diversity of learning experiences.
        """
        # Ensure prompt is a list, not a tensor
        if isinstance(prompt_tokens, torch.Tensor):
            prompt_tokens = prompt_tokens.tolist()
            
        stories = {}
        story_tokens = {}
        
        # Each agent writes a story
        for agent in self.agents:
            # Convert prompt to tensor for model input
            prompt_tensor = torch.tensor(prompt_tokens).long()
            
            # Generate longer response for stories
            response = agent.generate_response(prompt_tensor, max_length=30)
            response_tokens = response.tolist()
            stories[agent.name] = prompt_tokens + response_tokens
            story_tokens[agent.name] = response_tokens
        
        # Evaluate stories
        scores = {}
        for agent_name, story in stories.items():
            # For creative writing, we want diversity and coherence
            # Diversity: number of unique tokens
            unique_tokens = len(set(story))
            
            # Coherence: evaluate with feedback model
            coherence_score = self.evaluate_response_quality(story)
            
            # Combined score
            combined_score = 0.3 * (unique_tokens / len(story)) + 0.7 * coherence_score
            scores[agent_name] = combined_score
        
        # Determine best story
        best_writer = max(scores.items(), key=lambda x: x[1])[0]
        
        # Provide feedback to agents - normalize scores
        max_score = max(scores.values()) if scores else 1
        min_score = min(scores.values()) if scores else 0
        score_range = max(max_score - min_score, 0.001)  # Avoid division by zero
        
        for agent in self.agents:
            normalized_score = (scores[agent.name] - min_score) / score_range
            feedback = min(max(normalized_score, 0.0), 1.0)  # Ensure it stays in [0, 1]
            
            # Store experience with feedback
            agent.experiences.append((prompt_tokens, story_tokens[agent.name], feedback))
        
        # Record game outcome
        self.game_history.append({
            "type": "creative",
            "prompt": prompt_tokens,
            "stories": stories,
            "scores": scores,
            "best_writer": best_writer
        })
        
        return best_writer, scores
    
    def run_games(self, num_games, game_types=None):
        """Run multiple games of the specified types."""
        if game_types is None:
            game_types = ["debate", "qa", "creative"]
            
        results = {"wins": defaultdict(int), "scores": defaultdict(list)}
        
        for _ in range(num_games):
            # Select random game type
            game_type = random.choice(game_types)
            
            # Generate random topic/question/prompt
            tokens = [random.randint(1, self.vocab_size-1) for _ in range(random.randint(3, 7))]
            
            # Run appropriate game
            if game_type == "debate":
                winner, scores = self.run_debate_game(tokens)
            elif game_type == "qa":
                winner, scores = self.run_question_answering_game(tokens)
            else:  # creative
                winner, scores = self.run_creative_writing_game(tokens)
            
            # Record results
            results["wins"][winner] += 1
            for agent_name, score in scores.items():
                results["scores"][agent_name].append(score)
                
        return results

class SocraticLearningFramework:
    """
    Framework for implementing Socratic learning through language games.
    """
    def __init__(self, num_agents=3, vocab_size=1000, embedding_dim=64, hidden_dim=128, 
                 use_feedback_model=True, output_dir="results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print("Initializing agents...")
        # Initialize agents
        self.agents = [
            SocraticAgent(vocab_size, embedding_dim, hidden_dim, name=f"Agent_{i}")
            for i in range(num_agents)
        ]
        
        print("Initializing language game environment...")
        # Initialize language game environment
        self.language_game = LanguageGame(self.agents, vocab_size, use_feedback_model=use_feedback_model)
        
        # Environment parameters
        self.vocab_size = vocab_size
        self.cycles_completed = 0
        
        # Metrics for tracking emergent properties
        self.complexity_over_time = []
        self.diversity_over_time = []
        self.self_modifications = defaultdict(list)
    
    def measure_emergent_properties(self, game_history):
        """
        Measure emergent properties of the system.
        This helps track the development of capabilities beyond initial programming.
        """
        if not game_history:
            return 0, 0
        
        # Last N games to analyze
        recent_games = game_history[-20:] if len(game_history) > 20 else game_history
        
        # Measure complexity: average response length
        response_lengths = []
        for game in recent_games:
            if game['type'] == 'debate':
                for turn in game['turns']:
                    response_lengths.append(len(turn['response']))
            elif game['type'] == 'qa':
                for answer in game['answers'].values():
                    response_lengths.append(len(answer))
            else:  # creative
                for story in game['stories'].values():
                    response_lengths.append(len(story))
        
        avg_complexity = sum(response_lengths) / len(response_lengths) if response_lengths else 0
        
        # Measure diversity: unique token ratios
        all_tokens = []
        for game in recent_games:
            if game['type'] == 'debate':
                for turn in game['turns']:
                    all_tokens.extend(turn['response'])
            elif game['type'] == 'qa':
                for answer in game['answers'].values():
                    all_tokens.extend(answer)
            else:  # creative
                for story in game['stories'].values():
                    all_tokens.extend(story)
        
        token_diversity = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
        
        return avg_complexity, token_diversity
    
    def run_cycle(self, num_games=10, game_types=None, learning_phase_length=5):
        """
        Run a complete cycle of Socratic learning.
        """
        if game_types is None:
            game_types = ["debate", "qa", "creative"]
            
        # Run games
        results = self.language_game.run_games(num_games, game_types)
        
        # Learning phase - agents learn from their experiences
        for i in range(learning_phase_length):
            print(f"Learning phase iteration {i+1}/{learning_phase_length}")
            for agent in self.agents:
                loss = agent.learn_from_experience()
                if i == learning_phase_length - 1:  # Only on the last iteration
                    print(f"{agent.name} learning loss: {loss:.6f}")
        
        # Self-modification phase - agents attempt to improve their architecture
        for agent in self.agents:
            if random.random() < 0.3:  # 30% chance to attempt modification
                mod_type = random.choice(['scaling', 'revert'])
                success, message = agent.attempt_self_modification(mod_type)
                if success:
                    print(f"{agent.name} self-modification: {message}")
                    self.self_modifications[agent.name].append({
                        'cycle': self.cycles_completed,
                        'type': mod_type,
                        'message': message
                    })
        
        # Measure emergent properties
        complexity, diversity = self.measure_emergent_properties(self.language_game.game_history)
        self.complexity_over_time.append(complexity)
        self.diversity_over_time.append(diversity)
        
        # Track agent performances
        agent_performances = {}
        for agent in self.agents:
            if agent.performance_history:
                agent_performances[agent.name] = agent.performance_history[-1]
            
        self.cycles_completed += 1
        
        return {
            "cycle": self.cycles_completed,
            "results": results,
            "agent_performances": agent_performances,
            "complexity": complexity,
            "diversity": diversity
        }
    
    def run_recursive_improvement(self, num_cycles=5, games_per_cycle=20, learn_iterations=5):
        """
        Run multiple cycles to demonstrate recursive self-improvement.
        """
        all_results = []
        
        for cycle in range(num_cycles):
            print(f"\nRunning learning cycle {cycle+1}/{num_cycles}")
            cycle_result = self.run_cycle(num_games=games_per_cycle, learning_phase_length=learn_iterations)
            all_results.append(cycle_result)
            
            # Save intermediate results
            self.save_results(all_results, intermediate=True, cycle=cycle+1)
            
        return all_results
    
    def save_results(self, results, intermediate=False, cycle=None):
        """Save results and plots."""
        # Create timestamp for filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        suffix = f"_cycle{cycle}" if intermediate else "_final"
        
        # Save metrics plots
        self.plot_metrics(results, f"{self.output_dir}/metrics_{timestamp}{suffix}.png")
        
        # Save emergence metrics
        self.plot_emergence_metrics(f"{self.output_dir}/emergence_{timestamp}{suffix}.png")
        
        # Save modification history
        if self.self_modifications:
            self.plot_modifications(f"{self.output_dir}/modifications_{timestamp}{suffix}.png")
    
    def plot_metrics(self, results, filename):
        """Plot performance metrics."""
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Wins over time
        plt.subplot(2, 2, 1)
        wins_over_time = defaultdict(list)
