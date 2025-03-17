
# Generated code based on the paper
# Template: pytorch_model.py
# Subfield: artificial_intelligence
# Iteration: 3/3

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
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import copy
import time
import os
import json
from torch.utils.data import Dataset, DataLoader

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class AgentModel(nn.Module):
    """
    Neural language model for agents in the Socratic learning system.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1,
                model_type='lstm', attention_heads=None):
        super(AgentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.attention_heads = attention_heads
        
        if model_type == 'lstm':
            self.rnn = nn.LSTM(
                embedding_dim, 
                hidden_dim, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif model_type == 'gru':
            self.rnn = nn.GRU(
                embedding_dim, 
                hidden_dim, 
                num_layers=num_layers, 
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif model_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=attention_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.positional_encoding = PositionalEncoding(embedding_dim, dropout)
            self.hidden_proj = nn.Linear(embedding_dim, hidden_dim)
            
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None, attention_mask=None):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_len, embedding_dim)
        
        if self.model_type == 'transformer':
            # Add positional encoding
            embedded = self.positional_encoding(embedded)
            
            # Create padding mask if needed
            if attention_mask is None:
                attention_mask = (x != 0).float()
                attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
                attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))
            
            # Pass through transformer
            output = self.transformer_encoder(embedded, src_key_padding_mask=attention_mask)
            output = self.hidden_proj(output)
        else:
            # Pass through RNN
            if hidden is None:
                output, hidden = self.rnn(embedded)
            else:
                output, hidden = self.rnn(embedded, hidden)
        
        # Apply dropout
        output = self.dropout_layer(output)
        
        # Final projection
        output = self.fc(output)
        # output shape: (batch_size, seq_len, vocab_size)
        return output, hidden
    
    def predict_next_token(self, sequence, temperature=1.0, top_k=None):
        """Generate the next token given a sequence."""
        with torch.no_grad():
            output, _ = self.forward(sequence)
            # Get the last token prediction
            logits = output[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None and top_k > 0:
                top_k = min(top_k, logits.size(-1))
                # Zero out all values below the top-k values
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
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
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "model_type": self.model_type,
            "attention_heads": self.attention_heads
        }

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class MetaLearningModule(nn.Module):
    """
    Module that generates meta-parameters for self-modification.
    This is a key component for true recursive self-improvement.
    """
    def __init__(self, input_dim, hidden_dim):
        super(MetaLearningModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_arch = nn.Linear(hidden_dim // 2, 3)  # Architecture parameters (layers, hidden_dim, etc.)
        self.fc_lr = nn.Linear(hidden_dim // 2, 1)    # Learning rate
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # Generate architectural modification parameters
        arch_params = self.fc_arch(x)  # Raw parameters
        
        # Generate learning rate parameter (0.0001 to 0.01)
        lr = 0.0001 + 0.01 * self.sigmoid(self.fc_lr(x))
        
        return {
            'arch_params': arch_params,
            'learning_rate': lr
        }

class ExperienceBuffer:
    """
    Buffer to store and manage agent experiences.
    """
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size
        self.position = 0
        self.priorities = []
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling factor
        
    def add(self, experience, priority=None):
        if priority is None:
            # Default priority based on the feedback in the experience
            priority = experience[2] + 0.01  # Adding small constant to avoid zero priority
            
        if len(self.buffer) < self.max_size:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
            
        self.position = (self.position + 1) % self.max_size
        
    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
        
        # Convert priorities to probabilities
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()
        
        # Sample based on priorities
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), p=probs, replace=False)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        experiences = [self.buffer[idx] for idx in indices]
        return experiences, weights, indices
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            
    def __len__(self):
        return len(self.buffer)

class SocraticAgent:
    """
    Agent that can participate in language games and improve through Socratic learning.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, learning_rate=0.001, 
                 name="Agent", model_type='lstm', meta_learning=True):
        self.name = name
        self.model_type = model_type
        self.meta_learning = meta_learning
        
        # Initialize standard model
        if model_type == 'transformer':
            attention_heads = 4
            self.model = AgentModel(
                vocab_size, embedding_dim, hidden_dim, 
                num_layers=2, dropout=0.1, model_type=model_type, 
                attention_heads=attention_heads
            )
        else:
            self.model = AgentModel(
                vocab_size, embedding_dim, hidden_dim, 
                num_layers=2, dropout=0.1, model_type=model_type
            )
            
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        
        # Meta-learning module
        if meta_learning:
            self.meta_module = MetaLearningModule(hidden_dim, hidden_dim * 2)
            self.meta_optimizer = optim.Adam(self.meta_module.parameters(), lr=0.0005)
        
        # Experience management
        self.experience_buffer = ExperienceBuffer(max_size=5000)
        self.performance_history = []
        self.model_history = []
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        
        # Tracking improvements and capabilities
        self.skill_metrics = {
            'coherence': [],
            'diversity': [],
            'novelty': [],
            'complexity': []
        }
        self.training_iterations = 0
        
    def generate_response(self, prompt, max_length=20, temperature=0.8, top_k=50):
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
            next_token = self.model.predict_next_token(response, temperature, top_k)
            response = torch.cat([response, next_token], dim=1)
            
            # Optional: Early stopping if EOS token is generated
            if next_token.item() == 2:  # Assuming 2 is the EOS token
                break
            
        return response.squeeze(0)[len(prompt):]  # Return only the new tokens
    
    def learn_from_experience(self, batch_size=32, epochs=3):
        """Update model based on stored experiences."""
        if len(self.experience_buffer) < batch_size:
            return 0  # Not enough data to learn from
        
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for epoch in range(epochs):
            epoch_loss = 0
            experiences, weights, indices = self.experience_buffer.sample(batch_size * 4)
            
            # Process in batches
            for i in range(0, len(experiences), batch_size):
                batch = experiences[i:i+batch_size]
                batch_weights = weights[i:i+batch_size]
                batch_indices = indices[i:i+batch_size]
                
                # Process sequences
                inputs, targets, feedbacks = zip(*batch)
                
                # Convert tensors to lists if needed
                inputs = [input_seq.tolist() if isinstance(input_seq, torch.Tensor) else input_seq for input_seq in inputs]
                targets = [target_seq.tolist() if isinstance(target_seq, torch.Tensor) else target_seq for target_seq in targets]
                
                # Flatten nested lists if needed
                inputs = [input_seq[0] if isinstance(input_seq, list) and input_seq and isinstance(input_seq[0], list) else input_seq for input_seq in inputs]
                targets = [target_seq[0] if isinstance(target_seq, list) and target_seq and isinstance(target_seq[0], list) else target_seq for target_seq in targets]
                
                # Find max length for padding
                max_in_len = max(len(seq) for seq in inputs)
                max_target_len = max(len(seq) for seq in targets)
                
                # Pad sequences
                padded_inputs = [seq + [0] * (max_in_len - len(seq)) for seq in inputs]
                padded_targets = [seq + [0] * (max_target_len - len(seq)) for seq in targets]
                
                # Convert to tensors
                inputs_tensor = torch.tensor(padded_inputs).long()
                targets_tensor = torch.tensor(padded_targets).long()
                weights_tensor = torch.tensor(batch_weights).float() * torch.tensor(feedbacks).float()
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs, _ = self.model(inputs_tensor)
                
                # Calculate loss with importance sampling weights
                loss = 0
                new_priorities = []
                
                # Process each sequence individually
                for j, (output, target, weight) in enumerate(zip(outputs, targets_tensor, weights_tensor)):
                    target_len = len(targets[j])
                    valid_output = output[:target_len, :]
                    valid_target = target[:target_len]
                    
                    # Reshape for loss calculation
                    valid_output = valid_output.view(-1, self.vocab_size)
                    valid_target = valid_target.view(-1)
                    
                    # Calculate unweighted loss for priority update
                    with torch.no_grad():
                        unweighted_loss = self.criterion(valid_output, valid_target)
                        new_priorities.append(unweighted_loss.item() + 0.01)  # Avoid zero priority
                    
                    # Weight the loss by feedback and importance sampling
                    sample_loss = self.criterion(valid_output, valid_target) * weight
                    loss += sample_loss
                
                # Average loss across batch
                batch_loss = loss / len(inputs)
                
                # Backward pass
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # Gradient clipping
                self.optimizer.step()
                
                # Update priorities in experience buffer
                self.experience_buffer.update_priorities(batch_indices, new_priorities)
                
                epoch_loss += batch_loss.item()
                batch_count += 1
                
            # Update learning rate based on performance
            self.scheduler.step(epoch_loss)
            
        # Track performance history
        avg_loss = total_loss / max(1, batch_count)
        self.performance_history.append(avg_loss)
        self.training_iterations += 1
        
        # Meta-learning step: learn how to modify the model based on performance
        if self.meta_learning and len(self.performance_history) >= 3:
            self._meta_learning_step()
            
        return avg_loss
    
    def _meta_learning_step(self):
        """Meta-learning step to generate self-modification parameters."""
        if not hasattr(self, 'meta_module'):
            return
        
        # Create a performance vector as input to meta-module
        recent_perf = torch.tensor(self.performance_history[-10:])
        if len(recent_perf) < 10:
            recent_perf = F.pad(recent_perf, (0, 10 - len(recent_perf)))
            
        # Get embedding representations of recent responses
        recent_responses = []
        for exp in self.experience_buffer.buffer[-5:]:
            if not exp:
                continue
            target_tokens = exp[1]
            if isinstance(target_tokens, torch.Tensor):
                target_tokens = target_tokens.tolist()
            if not target_tokens:
                continue
            
            # Use model embedding to get representation
            with torch.no_grad():
                tokens_tensor = torch.tensor([target_tokens]).long()
                embedded = self.model.embedding(tokens_tensor)
                representation = embedded.mean(dim=1)  # Average embedding
                recent_responses.append(representation)
        
        # If we have recent responses, include their representations
        if recent_responses:
            response_tensor = torch.cat(recent_responses, dim=0)
            response_repr = response_tensor.mean(dim=0)
            
            # Combine with performance history
            input_tensor = torch.cat([response_repr, recent_perf.mean().unsqueeze(0)])
        else:
            # Fallback to just performance data
            zeros = torch.zeros(self.meta_module.fc1.in_features - 1)
            input_tensor = torch.cat([zeros, recent_perf.mean().unsqueeze(0)])
        
        # Generate modification parameters
        self.meta_optimizer.zero_grad()
        meta_params = self.meta_module(input_tensor.unsqueeze(0))
        
        # The loss here is based on expected future performance improvement
        # For simplicity, we use a proxy based on current trajectory
        perf_trajectory = (self.performance_history[-1] - self.performance_history[-2]) 
        if perf_trajectory >= 0:
            # Performance is getting worse (higher loss)
            # Encourage more aggressive model changes
            meta_loss = -torch.norm(meta_params['arch_params']) * 0.1
        else:
            # Performance is improving (lower loss)
            # Encourage more conservative model changes
            meta_loss = torch.norm(meta_params['arch_params']) * 0.1
            
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_params
    
    def save_model_checkpoint(self):
        """Save the current model state as a checkpoint."""
        checkpoint = {
            'model_state': copy.deepcopy(self.model.state_dict()),
            'config': self.model.get_config(),
            'performance': self.performance_history[-1] if self.performance_history else float('inf'),
            'skill_metrics': copy.deepcopy(self.skill_metrics)
        }
        self.model_history.append(checkpoint)
        
        # Keep only the best 3 models to save memory
        if len(self.model_history) > 3:
            self.model_history = sorted(self.model_history, key=lambda x: x['performance'])[:3]
    
    def attempt_self_modification(self, modification_type=None):
        """
        Attempt to modify the model architecture based on performance and meta-learning.
        This implements the self-referential capabilities described in the paper.
        """
        if len(self.performance_history) < 3:
            return False, "Not enough performance history"
        
        # Save current model before modification
        self.save_model_checkpoint()
        
        # Use meta-learning to suggest modifications if available
        if self.meta_learning and hasattr(self, 'meta_module'):
            meta_params = self._meta_learning_step()
            
            # Interpret meta parameters
            if meta_params is not None:
                arch_params = meta_params['arch_params'].detach().squeeze().tolist()
                # Convert to rounded integers for architecture changes
                layer_change = round(arch_params[0])  # -1, 0, or 1 to remove/keep/add layers
                hidden_scale = 1.0 + 0.2 * arch_params[1]  # Scale hidden dim up or down by up to 20%
                model_change = round(arch_params[2])  # -1, 0, 1 for LSTM, GRU, Transformer
                
                new_learning_rate = meta_params['learning_rate'].item()
                
                # Determine modification type based on meta parameters
                if abs(layer_change) > abs(hidden_scale - 1.0) and abs(layer_change) > abs(model_change):
                    modification_type = 'layers'
                elif abs(hidden_scale - 1.0) > abs(model_change):
                    modification_type = 'scaling'
                elif model_change != 0:
                    modification_type = 'architecture'
                else:
                    modification_type = 'learning_rate'
            else:
                # Fallback to random selection
                modification_type = random.choice(['scaling', 'layers', 'architecture', 'learning_rate', 'revert'])
        else:
            # Without meta-learning, choose randomly
            modification_type = random.choice(['scaling', 'layers', 'architecture', 'learning_rate', 'revert'])
            
        # Get current configuration
        current_config = self.model.get_config()
        
        # Execute the selected modification
        if modification_type == 'scaling':
            # Scale up or down hidden dimension based on recent performance
            scale_factor = 1.2 if self.performance_history[-1] < self.performance_history[-2] else 0.8
            new_hidden_dim = max(32, int(current_config['hidden_dim'] * scale_factor))
            
            # Create new model with adjusted capacity
            new_model = AgentModel(
                current_config['vocab_size'],
                current_config['embedding_dim'],
                new_hidden_dim,
                num_layers=current_config['num_layers'],
                dropout=current_config['dropout'],
                model_type=current_config['model_type'],
                attention_heads=current_config['attention_heads']
            )
            
            # Transfer knowledge where possible
            new_model.embedding.weight.data = self.model.embedding.weight.data.clone()
            
            # The output layer can be partially transferred
            min_dim = min(new_hidden_dim, current_config['hidden_dim'])
            new_model.fc.weight.data[:, :min_dim] = self.model.fc.weight.data[:, :min_dim].clone()
            new_model.fc.bias.data = self.model.fc.bias.data.clone()
            
            # Replace model and optimizer
            self.model = new_model
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
            
            return True, f"Scaled hidden dimension from {current_config['hidden_dim']} to {new_hidden_dim}"
            
        elif modification_type == 'layers':
            # Add or remove layers based on recent performance
            if self.performance_history[-1] < self.performance_history[-2]:
                # Performance is improving, add a layer if we're not already too deep
                new_layers = min(4, current_config['num_layers'] + 1)
            else:
                # Performance is degrading, remove a layer if possible
                new_layers = max(1, current_config['num_layers'] - 1)
                
            if new_layers == current_config['num_layers']:
                return False, "No change in number of layers"
                
            # Create new model with adjusted layers
            new_model = AgentModel(
                current_config['vocab_size'],
                current_config['embedding_dim'],
                current_config['hidden_dim'],
                num_layers=new_layers,
                dropout=current_config['dropout'],
                model_type=current_config['model_type'],
                attention_heads=current_config['attention_heads']
            )
            
            # Transfer embedding weights
            new_model.embedding.weight.data = self.model.embedding.weight.data.clone()
            
            # Transfer output layer weights
            new_model.fc.weight.data = self.model.fc.weight.data.clone()
            new_model.fc.bias.data = self.model.fc.bias.data.clone()
            
            # Replace model and optimizer
            self.model = new_model
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
            
            return True, f"Changed number of layers from {current_config['num_layers']} to {new_layers}"
            
        elif modification_type == 'architecture':
            # Change model architecture (LSTM, GRU, Transformer)
            current_type = current_config['model_type']
            model_types = ['lstm', 'gru']
            if current_config['embedding_dim'] % 2 == 0:  # Requirements for transformer
                model_types.append('transformer')
                
            # Remove current type and select a new one
            if current_type in model_types:
                model_types.remove(current_type)
            new_type = random.choice(model_types)
            
            # Adjust for transformer requirements
            attention_heads = None
            if new_type == 'transformer':
                attention_heads = 4
                
            # Create new model with different architecture
            new_model = AgentModel(
                current_config['vocab_size'],
                current_config['embedding_dim'],
                current_config['hidden_dim'],
                num_layers=current_config['num_layers'],
                dropout=current_config['dropout'],
                model_type=new_type,
                attention_heads=attention_heads
            )
            
            # Transfer embedding weights
            new_model.embedding.weight.data = self.model.embedding.weight.data.clone()
            
            # Output layer can be transferred
            new_model.fc.weight.data = self.model.fc.weight.data.clone()
            new_model.fc.bias.data = self.model.fc.bias.data.clone()
            
            # Replace model and optimizer
            self.model = new_model
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
            
            return True, f"Changed model architecture from {current_type} to {new_type}"
            
        elif modification_type == 'learning_rate':
            # Adjust learning rate based on recent performance
            if self.performance_history[-1] > self.performance_history[-2]:
                # Performance is getting worse, decrease learning rate
                new_lr = self.learning_rate * 0.5
            else:
                # Performance is improving, increase learning rate slightly
                new_lr = self.learning_rate * 1.2
                
            # Ensure learning rate is within reasonable bounds
            new_lr = max(0.00001, min(0.01, new_lr))
            
            # Update optimizer with new learning rate
            self.learning_rate = new_lr
            self.optimizer = optim.Adam(self.model.parameters(), lr=new_lr)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
            
            return True, f"Adjusted learning rate from {self.learning_rate:.6f} to {new_lr:.6f}"
            
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
                best_model['config']['hidden_dim'],
                num_layers=best_model['config']['num_layers'],
                dropout=best_model['config']['dropout'],
                model_type=best_model['config']['model_type'],
                attention_heads=best_model['config']['attention_heads']
            )
            
            # Load state from checkpoint
            new_model.load_state_dict(best_model['model_state'])
            
            # Replace model and optimizer
            self.model = new_model
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
            )
            
            # Also restore skill metrics
            self.skill_metrics = copy.deepcopy(best_model['skill_metrics'])
            
            return True, f"Reverted to best model with performance {best_model['performance']:.6f}"
            
        return False, "No modification made"
    
    def evaluate_skills(self, responses):
        """
        Evaluate agent's skills based on its recent responses.
        This provides metrics to track progression of capabilities.
        """
        if not responses:
            return
            
        # Coherence: ratio of duplicate adjacent tokens (less is better)
        coherence_scores = []
        for response in responses:
            if len(response) <= 1:
                continue
            
