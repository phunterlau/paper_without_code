import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import math
import random
import sympy as sp

import json
import argparse
import os
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

class LyapunovSystem:
    """
    Represents a dynamical system with its Lyapunov function.
    Supports both symbolic and numerical evaluation.
    """
    def __init__(self, system_eqs: List[str], lyap_func: Optional[str] = None):
        # Input validation
        if not isinstance(system_eqs, list):
            raise TypeError("system_eqs must be a list of strings")
            
        self.system_eqs = system_eqs
        self.lyap_func = lyap_func
        self.dim = len(system_eqs)
        
        # Create symbolic variables
        self.vars = sp.symbols(f'x:{self.dim}')
        
        # Parse equations
        try:
            self.system_sym = [sp.sympify(eq) for eq in system_eqs]
            if lyap_func:
                self.lyap_sym = sp.sympify(lyap_func)
        except sp.SympifyError as e:
            raise ValueError(f"Failed to parse equations: {e}")
    
    def evaluate_system(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the system equations at point x"""
        if len(x) != self.dim:
            raise ValueError(f"Expected input of dimension {self.dim}, got {len(x)}")
            
        subs_dict = {var: val for var, val in zip(self.vars, x)}
        return np.array([float(eq.evalf(subs=subs_dict)) for eq in self.system_sym])
    
    def evaluate_lyapunov(self, x: np.ndarray) -> float:
        """Evaluates the Lyapunov function at point x"""
        if not self.lyap_func:
            raise ValueError("No Lyapunov function defined")
            
        if len(x) != self.dim:
            raise ValueError(f"Expected input of dimension {self.dim}, got {len(x)}")
            
        subs_dict = {var: val for var, val in zip(self.vars, x)}
        return float(self.lyap_sym.evalf(subs=subs_dict))

    def verify_lyapunov(self, grid_points: int = 20, bounds: Tuple[float, float] = (-2, 2)) -> bool:
        """
        Verifies Lyapunov function properties:
        1. V(0) = 0
        2. V(x) > 0 for x ≠ 0
        3. ∇V·f ≤ 0
        """
        if not self.lyap_func:
            return False
            
        try:
            # Check V(0) = 0
            if abs(self.evaluate_lyapunov(np.zeros(self.dim))) > 1e-10:
                print("Failed: V(0) ≠ 0")
                return False
                
            # Generate grid points for verification
            grid = np.linspace(bounds[0], bounds[1], grid_points)
            points = np.array(np.meshgrid(*[grid for _ in range(self.dim)]))
            points = points.reshape(self.dim, -1).T
            
            for x in points:
                if np.linalg.norm(x) > 1e-10:  # Skip origin
                    # Check V(x) > 0
                    v_x = self.evaluate_lyapunov(x)
                    if v_x <= 0:
                        print(f"Failed: V(x) ≤ 0 at x = {x}, V(x) = {v_x}")
                        return False
                    
                    # Check ∇V·f ≤ 0
                    grad_V = np.array([float(sp.diff(self.lyap_sym, var).evalf(subs={
                        v: val for v, val in zip(self.vars, x)})) for var in self.vars])
                    f_x = self.evaluate_system(x)
                    dot_product = np.dot(grad_V, f_x)
                    if dot_product > 1e-10:
                        print(f"Failed: ∇V·f > 0 at x = {x}, value = {dot_product}")
                        return False
                        
            return True
            
        except Exception as e:
            print(f"Verification failed with error: {e}")
            return False

class ForwardGenerator:
    """
    Implements forward generation of stable polynomial systems
    with stronger stability guarantees
    """
    def __init__(self, max_degree: int = 3):
        self.max_degree = max_degree
    
    def generate_hurwitz_matrix(self, dim: int) -> np.ndarray:
        """
        Generates a Hurwitz matrix (all eigenvalues have negative real parts)
        using the method from Chen & Zhou (1998)
        """
        # Generate random negative eigenvalues
        eigenvals = -np.random.uniform(0.5, 2.0, dim)
        # Create diagonal matrix
        D = np.diag(eigenvals)
        # Generate random orthogonal matrix
        Q = np.random.randn(dim, dim)
        Q, _ = np.linalg.qr(Q)  # QR decomposition gives orthogonal matrix
        # Create Hurwitz matrix
        A = Q @ D @ Q.T
        return A
    
    def generate_polynomial_system(self, dim: int) -> List[str]:
        """
        Generates a random polynomial system with stronger stability properties
        """
        # Generate Hurwitz linear part
        A = self.generate_hurwitz_matrix(dim)
        
        system = []
        for i in range(dim):
            terms = []
            
            # Add linear terms (Hurwitz part)
            for j in range(dim):
                if abs(A[i,j]) > 1e-10:
                    terms.append(f"{A[i,j]}*x{j}")
            
            # Add some higher-order terms with very small coefficients
            num_nonlin_terms = random.randint(0, 2)
            for _ in range(num_nonlin_terms):
                # Generate term with very small coefficient
                coeff = random.uniform(-0.1, 0.1)  # Reduced coefficient range
                if abs(coeff) < 1e-10:
                    continue
                
                # Generate powers ensuring total degree is limited
                powers = [0] * dim
                # Prefer terms that include the current variable
                if random.random() < 0.7:
                    powers[i] = random.randint(1, self.max_degree-1)
                # Add at most one other variable
                other_var = random.randint(0, dim-1)
                if other_var != i:
                    powers[other_var] = random.randint(1, self.max_degree-powers[i])
                
                if all(p == 0 for p in powers):
                    continue
                
                term = str(coeff)
                for var, power in enumerate(powers):
                    if power > 0:
                        term += f"*x{var}"
                        if power > 1:
                            term += f"**{power}"
                terms.append(term)
            
            system.append(" + ".join(terms) if terms else "0")
        
        return system
    
    def try_find_lyapunov_candidate(self, dim: int) -> np.ndarray:
        """Generate a candidate positive definite quadratic form"""
        # Generate random positive definite matrix with controlled condition number
        while True:
            Q = np.random.randn(dim, dim)
            Q = Q.T @ Q
            # Add scaled identity to improve conditioning
            scale = np.trace(Q) / dim
            Q = Q + np.eye(dim) * scale
            # Check condition number
            cond = np.linalg.cond(Q)
            if cond < 100:  # Ensure well-conditioned matrix
                break
        return Q
    
    def try_find_lyapunov(self, system_eqs: List[str], max_attempts: int = 5) -> Optional[str]:
        """
        Attempts to find a quadratic Lyapunov function with progressive verification
        """
        dim = len(system_eqs)
        
        # Try different strategies
        strategies = [
            (10, (-1.0, 1.0)),    # Coarse grid, small region
            (15, (-1.5, 1.5)),    # Medium grid, medium region
            (20, (-2.0, 2.0))     # Fine grid, full region
        ]
        
        for attempt in range(max_attempts):
            print(f"Attempt {attempt + 1}/{max_attempts} to find Lyapunov function...")
            
            # Try different scaling factors for the quadratic form
            scale = 1.0 / (attempt + 1)
            Q = self.try_find_lyapunov_candidate(dim) * scale
            
            # Construct Lyapunov function
            terms = []
            for i in range(dim):
                for j in range(i, dim):
                    coeff = Q[i,j]
                    if abs(coeff) < 1e-10:
                        continue
                    if i == j:
                        terms.append(f"{coeff}*x{i}**2")
                    else:
                        terms.append(f"{2*coeff}*x{i}*x{j}")
            
            lyap_func = " + ".join(terms)
            
            try:
                # Progressive verification
                system = LyapunovSystem(system_eqs, lyap_func)
                
                # Try verification with increasingly strict requirements
                is_valid = True
                for grid_points, bounds in strategies:
                    if not system.verify_lyapunov(grid_points=grid_points, bounds=bounds):
                        is_valid = False
                        break
                
                if is_valid:
                    print("Found valid Lyapunov function!")
                    return lyap_func
                    
            except Exception as e:
                print(f"Verification failed on attempt {attempt + 1}: {e}")
                continue
        
        print("Failed to find valid Lyapunov function")
        return None

class BackwardGenerator:
    """
    Implements backward generation starting from a known Lyapunov function.
    """
    def __init__(self, max_degree: int = 3):
        self.max_degree = max_degree
    
    def generate_example(self, dim: int) -> Tuple[List[str], str]:
        """
        Generates a Lyapunov function first, then constructs a compatible system.
        Following Section 4.1 of the paper.
        """
        # Generate V_proper (positive definite quadratic form)
        A = np.random.randn(dim, dim)
        A = A.T @ A + np.eye(dim)
        
        # Construct Lyapunov function
        terms = []
        for i in range(dim):
            for j in range(i, dim):
                coeff = A[i,j]
                if i == j:
                    terms.append(f"{coeff}*x{i}**2")
                else:
                    terms.append(f"{2*coeff}*x{i}*x{j}")
        lyap_func = " + ".join(terms)
        
        # Generate system equations that make this a Lyapunov function
        system = []
        for i in range(dim):
            # Negative gradient term ensures V'(x)f(x) ≤ 0
            grad_term = f"-{A[i,i]}*x{i}"
            for j in range(dim):
                if i != j:
                    grad_term += f" - {A[i,j]}*x{j}"
            
            # Add some cross terms (as in paper)
            num_cross = random.randint(0, 2)
            cross_terms = []
            for _ in range(num_cross):
                j = random.randint(0, dim-1)
                k = random.randint(0, dim-1)
                coeff = random.randint(-5, 5)
                if coeff != 0:
                    cross_terms.append(f"{coeff}*x{j}*x{k}")
            
            eq = " + ".join([grad_term] + cross_terms)
            system.append(eq)
        
        return system, lyap_func

class LyapunovDataset(Dataset):
    """
    Enhanced dataset class with proper loading support
    """
    def __init__(self, 
                 num_backward: int = 1000,
                 num_forward: int = 300,
                 dim: int = 2,
                 max_degree: int = 2,
                 max_seq_length: int = 100,
                 load_from_data: bool = False):
        self.max_seq_length = max_seq_length
        self.systems = []
        self.lyap_funcs = []
        
        # Special tokens
        self.special_tokens = {
            'NUM': '<NUM>',    # Represents any number
            'VAR': '<VAR>',    # Represents any variable
            'OP': '<OP>',      # Represents any operator
            'PAD': '<PAD>'     # Padding token
        }
        
        # Initialize basic vocabulary
        self.vocab = set(self.special_tokens.values())
        self.operators = {'+', '-', '*', '**'}
        self.vocab.update(self.operators)
        
        # Add variables up to max dimension
        for i in range(10):  # Support up to 10 dimensions
            self.vocab.add(f'x{i}')
            
        # If not loading from data, generate new examples
        if not load_from_data:
            # Generate backward examples
            print("Generating backward examples...")
            backward_gen = BackwardGenerator(max_degree=max_degree)
            for i in range(num_backward):
                if i % 100 == 0:
                    print(f"Generating backward example {i}/{num_backward}")
                system, lyap = backward_gen.generate_example(dim=dim)
                self.systems.append(system)
                self.lyap_funcs.append(lyap)
                
            # Generate forward examples
            print("\nGenerating forward examples...")
            forward_gen = ForwardGenerator(max_degree=max_degree)
            count = 0
            attempts = 0
            while count < num_forward and attempts < num_forward * 10:
                attempts += 1
                if attempts % 10 == 0:
                    print(f"Generated {count}/{num_forward} forward examples (attempts: {attempts})")
                
                system = forward_gen.generate_polynomial_system(dim=dim)
                lyap = forward_gen.try_find_lyapunov(system)
                if lyap is not None:
                    self.systems.append(system)
                    self.lyap_funcs.append(lyap)
                    count += 1
            
            print(f"\nFinal dataset size: {len(self.systems)} examples")
        
        # Update vocabulary with tokens from systems and functions
        self._update_vocabulary()
        
    def _update_vocabulary(self):
        """Update vocabulary from systems and Lyapunov functions"""
        for system in self.systems:
            for eq in system:
                tokens = eq.replace('+', ' + ').replace('-', ' - ').replace('*', ' * ').split()
                self.vocab.update(tokens)
        for lyap in self.lyap_funcs:
            tokens = lyap.replace('+', ' + ').replace('-', ' - ').replace('*', ' * ').split()
            self.vocab.update(tokens)
        
        self.token2idx = {token: idx for idx, token in enumerate(sorted(self.vocab))}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
    
    def __len__(self):
        return len(self.systems)
    
    @classmethod
    def from_saved_data(cls, systems, lyap_funcs, max_seq_length=100):
        """Create dataset from saved data"""
        dataset = cls(num_backward=0, num_forward=0, load_from_data=True)
        dataset.systems = systems
        dataset.lyap_funcs = lyap_funcs
        dataset.max_seq_length = max_seq_length
        dataset._update_vocabulary()
        return dataset
        
    def normalize_number(self, num_str: str) -> str:
        """Convert number to a standard format"""
        try:
            num = float(num_str)
            if abs(num - round(num)) < 1e-10:
                return str(round(num))
            return f"{num:.6f}"
        except ValueError:
            return num_str
    
    def tokenize_term(self, term: str) -> List[str]:
        """Tokenize a single term of the equation"""
        # Split around operators while keeping them
        tokens = []
        current_token = ""
        
        i = 0
        while i < len(term):
            if term[i] in '+-*':
                if current_token:
                    tokens.append(current_token)
                    current_token = ""
                if i + 1 < len(term) and term[i:i+2] == '**':
                    tokens.append('**')
                    i += 2
                else:
                    tokens.append(term[i])
                    i += 1
            else:
                current_token += term[i]
                i += 1
        
        if current_token:
            tokens.append(current_token)
        
        # Process each token
        processed_tokens = []
        for token in tokens:
            if token in self.operators:
                processed_tokens.append(token)
            elif token.startswith('x'):
                processed_tokens.append(token)
            else:
                try:
                    float(token)  # Check if it's a number
                    processed_tokens.append(self.special_tokens['NUM'])
                except ValueError:
                    processed_tokens.append(token)
        
        return processed_tokens
    
    def tokenize_system(self, system: List[str]) -> torch.Tensor:
        """Convert system equations to tensor format with improved tokenization"""
        all_tokens = []
        
        for eq in system:
            # Split equation into terms
            terms = eq.replace(' ', '').split('+')
            for term in terms:
                if term.startswith('-'):
                    all_tokens.append('-')
                    term = term[1:]
                all_tokens.extend(self.tokenize_term(term))
            all_tokens.append(self.special_tokens['PAD'])
        
        # Create encoding
        encoding = torch.zeros(self.max_seq_length, len(self.vocab))
        for i, token in enumerate(all_tokens[:self.max_seq_length]):
            if token in self.token2idx:
                encoding[i, self.token2idx[token]] = 1
            else:
                # Use special token if not in vocabulary
                if token.startswith('x'):
                    encoding[i, self.token2idx[self.special_tokens['VAR']]] = 1
                elif any(op in token for op in self.operators):
                    encoding[i, self.token2idx[self.special_tokens['OP']]] = 1
                else:
                    try:
                        float(token)
                        encoding[i, self.token2idx[self.special_tokens['NUM']]] = 1
                    except ValueError:
                        continue
        
        return encoding
    
    def tokenize_lyapunov(self, lyap: str) -> torch.Tensor:
        """Convert Lyapunov function to tensor format with improved tokenization"""
        tokens = self.tokenize_term(lyap.replace(' ', ''))
        
        encoding = torch.zeros(len(self.vocab))
        for token in tokens:
            if token in self.token2idx:
                encoding[self.token2idx[token]] = 1
            else:
                # Use special token if not in vocabulary
                if token.startswith('x'):
                    encoding[self.token2idx[self.special_tokens['VAR']]] = 1
                elif any(op in token for op in self.operators):
                    encoding[self.token2idx[self.special_tokens['OP']]] = 1
                else:
                    try:
                        float(token)
                        encoding[self.token2idx[self.special_tokens['NUM']]] = 1
                    except ValueError:
                        continue
        
        return encoding
        
    def __getitem__(self, idx):
        system = self.systems[idx]
        lyap = self.lyap_funcs[idx]
        
        # Convert to tensor format
        system_tensor = self.tokenize_system(system)
        lyap_tensor = self.tokenize_lyapunov(lyap)
        
        return system_tensor, lyap_tensor

class LyapunovDataManager:
    """
    Handles saving and loading of generated Lyapunov systems and functions
    """
    def __init__(self, save_dir: str = "saved_data"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save_dataset(self, dataset: LyapunovDataset, filename: str = None) -> str:
        """
        Saves a dataset to a JSON file
        Returns the filename used
        """
        if filename is None:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"lyapunov_data_{timestamp}.json"
        
        filepath = self.save_dir / filename
        
        # Prepare data for saving
        data = {
            'systems': dataset.systems,
            'lyap_funcs': dataset.lyap_funcs,
            'max_seq_length': dataset.max_seq_length,
            'metadata': {
                'num_examples': len(dataset),
                'vocab_size': len(dataset.vocab),
                'timestamp': datetime.now().isoformat(),
            }
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Dataset saved to {filepath}")
        return filename
    
    def load_dataset(self, filename: str) -> LyapunovDataset:
        """
        Loads a dataset from a JSON file
        """
        filepath = self.save_dir / filename
        
        # Load from file
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Create dataset using class method
        dataset = LyapunovDataset.from_saved_data(
            systems=data['systems'],
            lyap_funcs=data['lyap_funcs'],
            max_seq_length=data.get('max_seq_length', 100)
        )
        
        print(f"Loaded dataset from {filepath}")
        print(f"Number of examples: {len(dataset)}")
        print(f"Vocabulary size: {len(dataset.vocab)}")
        
        return dataset
    
    def list_saved_datasets(self) -> List[str]:
        """Returns list of saved dataset filenames"""
        return [f.name for f in self.save_dir.glob("*.json")]

class PositionalEncoding(nn.Module):
    """
    Positional encoding with proper device handling
    """
    def __init__(self, d_model: int, max_len: int = 5000, device=None):
        super().__init__()
        self.device = device
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Ensure pe is on the same device as x
        if self.pe.device != x.device:
            self.pe = self.pe.to(x.device)
        return x + self.pe[:, :x.size(1)]


class LyapunovLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        # Basic MSE loss
        mse_loss = self.mse(pred, target)
        
        # Additional penalty for missing quadratic terms
        quad_penalty = 0
        pred_probs = torch.softmax(pred, dim=-1)
        
        # Check for x0**2 and x1**2 terms
        x0_quad_mask = (target.sum(dim=1) > 0) & (pred_probs[:, target == 1].sum(dim=1) < 0.1)
        x1_quad_mask = (target.sum(dim=1) > 0) & (pred_probs[:, target == 1].sum(dim=1) < 0.1)
        
        quad_penalty = torch.mean(x0_quad_mask.float() + x1_quad_mask.float())
        
        return mse_loss + 0.5 * quad_penalty

class LyapunovTransformer(nn.Module):
    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 device=None):
        super().__init__()
        
        # Set device
        self.device = device if device is not None else \
            torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Initializing model on device: {self.device}")
        
        # Input embedding with scaling
        self.input_embedding = nn.Linear(input_size, d_model)
        self.embedding_scale = math.sqrt(d_model)
        self.pos_encoder = PositionalEncoding(d_model, device=self.device)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            device=self.device
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output generation
        self.output_projection = nn.Linear(d_model, dim_feedforward)
        self.output_activation = nn.GELU()
        self.output_dropout = nn.Dropout(dropout)
        self.term_generator = nn.Linear(dim_feedforward, output_size)
        
        # Move all components to device
        self.to(self.device)
    
    def forward(self, src, src_mask=None):
        # Ensure input is on correct device
        src = src.to(self.device)
        
        # If mask is provided, move it to device
        if src_mask is not None:
            src_mask = src_mask.to(self.device)
        
        # Process through model
        x = self.input_embedding(src) * self.embedding_scale
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_mask)
        x = self.output_projection(x)
        x = self.output_activation(x)
        x = self.output_dropout(x)
        x = self.term_generator(x)
        
        return x.mean(dim=1)
    
    def move_to_device(self, device=None):
        """Explicitly move model to specified device"""
        if device is not None:
            self.device = device
        self.to(self.device)
        return self

class LyapunovTrainer:
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 lr: float = 1e-4,
                 save_dir: str = "saved_models"):
        self.device = get_device()
        print(f"Trainer using device: {self.device}")
        
        self.model = model.move_to_device(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, epoch: int, val_loss: float, filename: str = None):
        """Save model checkpoint"""
        if filename is None:
            filename = f"model_epoch{epoch:03d}_loss{val_loss:.6f}.pt"
        
        filepath = self.save_dir / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, filepath)
        print(f"\nSaved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        return checkpoint['epoch']
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        with tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc="Training") as pbar:
            for batch_idx, (systems, targets) in pbar:
                systems = systems.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(systems)
                
                # Use custom loss
                loss = self.criterion(output, targets)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        if not self.val_loader:
            return None
            
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for systems, targets in tqdm(self.val_loader, desc="Validating", leave=False):
                systems = systems.to(self.device)
                targets = targets.to(self.device)
                
                output = self.model(systems)
                loss = self.criterion(output, targets)
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs: int, save_freq: int = 5):
        print(f"\nTraining on device: {self.device}")
        print(f"Training set size: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation set size: {len(self.val_loader.dataset)}")
        
        # Initialize plot
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Progress')
        
        train_line, = ax.plot([], [], label='Training Loss')
        if self.val_loader:
            val_line, = ax.plot([], [], label='Validation Loss')
        ax.legend()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            if self.val_loader:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                self.scheduler.step(val_loss)
                
                print(f'Epoch {epoch+1}, Train Loss: {train_loss:.6f}, '
                      f'Val Loss: {val_loss:.6f}')
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch + 1, val_loss, 'best_model.pt')
            else:
                print(f'Epoch {epoch+1}, Train Loss: {train_loss:.6f}')
            
            # Save checkpoint periodically
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(epoch + 1, val_loss if self.val_loader else train_loss)
            
            # Update plot
            train_line.set_data(range(1, len(self.train_losses) + 1), self.train_losses)
            if self.val_loader:
                val_line.set_data(range(1, len(self.val_losses) + 1), self.val_losses)
            
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.1)
        
        plt.ioff()
        self.plot_training_history()
    
    def plot_training_history(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.savefig(self.save_dir / 'training_history.png')
        plt.close()

def prepare_data_for_training(systems: List[str], lyap_funcs: List[str], 
                            max_seq_length: int = 100):
    """
    Converts symbolic expressions to tensor format for training
    """
    # Create vocabulary from all unique tokens
    vocab = set()
    for system in systems:
        for eq in system:
            tokens = eq.replace('+', ' + ').replace('-', ' - ').replace('*', ' * ').split()
            vocab.update(tokens)
    for lyap in lyap_funcs:
        tokens = lyap.replace('+', ' + ').replace('-', ' - ').replace('*', ' * ').split()
        vocab.update(tokens)
    
    # Create token to index mapping
    token2idx = {token: idx for idx, token in enumerate(sorted(vocab))}
    
    # Convert systems and Lyapunov functions to tensors
    X = torch.zeros(len(systems), max_seq_length, len(token2idx))
    y = torch.zeros(len(lyap_funcs), len(token2idx))
    
    for i, system in enumerate(systems):
        tokens = []
        for eq in system:
            tokens.extend(eq.replace('+', ' + ').replace('-', ' - ').replace('*', ' * ').split())
        for j, token in enumerate(tokens[:max_seq_length]):
            X[i, j, token2idx[token]] = 1
            
    for i, lyap in enumerate(lyap_funcs):
        tokens = lyap.replace('+', ' + ').replace('-', ' - ').replace('*', ' * ').split()
        for token in tokens:
            y[i, token2idx[token]] = 1
            
    return X, y, token2idx

def test_model(model: nn.Module, 
               dataset: LyapunovDataset, 
               num_tests: int = 5,
               save_dir: Optional[str] = None):
    """
    Enhanced test function with better prediction validation
    """
    device = next(model.parameters()).device
    model.eval()
    results = {
        'backward': {'success': 0, 'total': 0},
        'forward': {'success': 0, 'total': 0}
    }
    
    def verify_prediction(system: List[str], pred_lyap: str) -> bool:
        """Enhanced verification of predicted Lyapunov function"""
        try:
            # Basic syntax check
            if not all(term in pred_lyap for term in ['x0**2', 'x1**2']):
                print("Prediction missing required quadratic terms")
                return False
            
            # Parse and normalize coefficients
            terms = pred_lyap.split('+')
            normalized_terms = []
            for term in terms:
                term = term.strip()
                if not any(c.isdigit() for c in term):
                    term = '1.0*' + term
                normalized_terms.append(term)
            
            normalized_lyap = ' + '.join(normalized_terms)
            
            # Create and verify system
            system_obj = LyapunovSystem(system, normalized_lyap)
            return system_obj.verify_lyapunov(grid_points=30)  # Increased verification points
            
        except Exception as e:
            print(f"Verification failed: {e}")
            return False
    
    with torch.no_grad():
        # Test on backward generated examples
        print("\nTesting on backward generated examples:")
        backward_gen = BackwardGenerator()
        for i in range(num_tests):
            system, true_lyap = backward_gen.generate_example(dim=2)
            system_tensor = dataset.tokenize_system(system)
            system_tensor = system_tensor.unsqueeze(0).to(device)
            
            # Get prediction with temperature scaling
            logits = model(system_tensor)
            pred = decode_prediction(logits, dataset.idx2token)
            
            print(f"\nTest {i+1}:")
            print(f"System: {system}")
            print(f"True Lyapunov: {true_lyap}")
            print(f"Predicted: {pred}")
            
            is_valid = verify_prediction(system, pred)
            results['backward']['total'] += 1
            if is_valid:
                results['backward']['success'] += 1
                print("Prediction is valid!")
            else:
                print("Prediction is invalid.")
        
        # Test on forward generated examples
        print("\nTesting on forward generated examples:")
        forward_gen = ForwardGenerator()
        for i in range(num_tests):
            system = forward_gen.generate_polynomial_system(dim=2)
            true_lyap = forward_gen.try_find_lyapunov(system)
            
            if true_lyap:
                system_tensor = dataset.tokenize_system(system)
                system_tensor = system_tensor.unsqueeze(0).to(device)
                
                pred = model(system_tensor)
                pred_lyap = decode_prediction(pred, dataset.idx2token)
                
                print(f"\nTest {i+1}:")
                print(f"System: {system}")
                print(f"True Lyapunov: {true_lyap}")
                print(f"Predicted: {pred_lyap}")
                
                is_valid = verify_prediction(system, pred_lyap)
                results['forward']['total'] += 1
                if is_valid:
                    results['forward']['success'] += 1
                    print("Prediction is valid!")
                else:
                    print("Prediction is invalid.")
    
    # Print summary
    print("\nTest Results:")
    print(f"Backward generation: {results['backward']['success']}/{results['backward']['total']} "
          f"({results['backward']['success']/results['backward']['total']*100:.1f}%)")
    if results['forward']['total'] > 0:
        print(f"Forward generation: {results['forward']['success']}/{results['forward']['total']} "
              f"({results['forward']['success']/results['forward']['total']*100:.1f}%)")
    else:
        print("No forward generation tests completed")
    
    return results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Lyapunov Function Discovery Training')
    parser.add_argument('--load', type=str, help='Load saved dataset from file')
    parser.add_argument('--save', type=str, help='Save generated dataset to file')
    parser.add_argument('--list', action='store_true', help='List saved datasets')
    parser.add_argument('--num-backward', type=int, default=1000, 
                       help='Number of backward generated examples')
    parser.add_argument('--num-forward', type=int, default=300,
                       help='Number of forward generated examples')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()

    # Set device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize data manager
    data_manager = LyapunovDataManager()
    
    # List saved datasets if requested
    if args.list:
        print("\nSaved datasets:")
        for filename in data_manager.list_saved_datasets():
            print(f"  {filename}")
        return
    
    # Load or generate dataset
    if args.load:
        print(f"\nLoading dataset from {args.load}")
        dataset = data_manager.load_dataset(args.load)
    else:
        print("\nGenerating new dataset...")
        dataset = LyapunovDataset(
            num_backward=args.num_backward,
            num_forward=args.num_forward
        )
        
        # Save dataset if requested
        if args.save:
            data_manager.save_dataset(dataset, args.save)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model
    print("\nInitializing model...")
    vocab_size = len(dataset.vocab)
    model = LyapunovTransformer(
        input_size=vocab_size,
        output_size=vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        device=device
    )
    
    # Train model
    print("\nStarting training...")
    trainer = LyapunovTrainer(model, train_loader, val_loader)
    trainer.train(num_epochs=args.epochs)
    
    # Test model
    print("\nTesting model on examples...")
    test_results = test_model(model, dataset, num_tests=5)
    
    # Save test results if saving directory is specified
    if args.save:
        save_path = Path(args.save).parent / 'test_results.json'
        with open(save_path, 'w') as f:
            json.dump(test_results, f, indent=2)

def decode_prediction(pred: torch.Tensor, idx2token: dict) -> str:
    """
    Converts model output back to symbolic expression
    """
    tokens = []
    for idx in pred.argmax(dim=1):
        tokens.append(idx2token[idx.item()])
    print(f"decoded tokens: {tokens}")
    return ' '.join(tokens)

def test_decode_prediction():
    """
    Test function to verify decoder output format
    """
    # Create dummy logits
    logits = torch.randn(100)  # Adjust size based on vocabulary
    
    # Create dummy idx2token mapping
    idx2token = {
        i: str(i/10) for i in range(100)  # Dummy tokens including some that look like numbers
    }
    
    # Test decode
    result = decode_prediction(logits, idx2token)
    
    print(f"Generated Lyapunov function: {result}")
    
    # Verify format
    try:
        import sympy as sp
        x0, x1 = sp.symbols('x0 x1')
        expr = sp.sympify(result)
        print("Successfully parsed expression!")
        return True
    except Exception as e:
        print(f"Failed to parse expression: {e}")
        return False

def get_device(device_arg=None):
    """
    Enhanced device selection with MPS support
    """
    if device_arg is not None:
        return torch.device(device_arg)
    
    # Check for MPS (Mac GPU)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Using MPS (Mac GPU) device")
        return torch.device('mps')
    # Check for CUDA
    elif torch.cuda.is_available():
        print("Using CUDA device")
        return torch.device('cuda')
    # Fall back to CPU
    else:
        print("Using CPU device")
        return torch.device('cpu')

if __name__ == "__main__":
    main()
