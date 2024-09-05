import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from io import StringIO
from Bio import SeqIO
from Bio.PDB import PDBParser
from rdkit import Chem
from matplotlib import pyplot as plt

# Hard-coded PDB data (simplified Crambin structure)
PDB_DATA = """
ATOM      1  N   THR A   1      17.047  14.099   3.625  1.00 13.79      1CRN N
ATOM      2  CA  THR A   1      16.967  12.784   4.338  1.00 10.80      1CRN C
ATOM      3  C   THR A   1      15.685  12.755   5.133  1.00  9.19      1CRN C
ATOM      4  O   THR A   1      15.268  13.825   5.594  1.00  9.85      1CRN O
ATOM      5  N   THR A   2      15.115  11.555   5.265  1.00  7.81      1CRN N
ATOM      6  CA  THR A   2      13.856  11.469   6.066  1.00  8.31      1CRN C
ATOM      7  C   THR A   2      14.164  10.785   7.379  1.00  5.80      1CRN C
ATOM      8  O   THR A   2      14.993   9.862   7.443  1.00  6.94      1CRN O
ATOM      9  N   CYS A   3      13.488  11.241   8.417  1.00  5.24      1CRN N
ATOM     10  CA  CYS A   3      13.660  10.707   9.787  1.00  5.39      1CRN C
ATOM     11  C   CYS A   3      12.269  10.431  10.323  1.00  4.45      1CRN C
ATOM     12  O   CYS A   3      11.325  11.161  10.185  1.00  5.05      1CRN O
ATOM     13  N   CYS A   4      12.019   9.354  11.085  1.00  3.90      1CRN N
ATOM     14  CA  CYS A   4      10.646   9.093  11.640  1.00  4.24      1CRN C
ATOM     15  C   CYS A   4      10.654   9.329  13.139  1.00  3.72      1CRN C
ATOM     16  O   CYS A   4      11.659   9.296  13.850  1.00  4.13      1CRN O
ATOM     17  N   PRO A   5       9.561   9.677  13.604  1.00  3.96      1CRN N
ATOM     18  CA  PRO A   5       9.448  10.102  15.035  1.00  4.25      1CRN C
ATOM     19  C   PRO A   5      10.000   9.130  16.069  1.00  4.27      1CRN C
ATOM     20  O   PRO A   5       9.685   9.241  17.253  1.00  4.94      1CRN O
END
"""

# Hard-coded MSA data in FASTA format
MSA_DATA = """
>1CRN:A|PDBID|CHAIN|SEQUENCE
TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN
>seq1
TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN
>seq2
TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN
>seq3
TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN
>seq4
TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN
>seq5
TTCCPSIVARSDFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN
"""

def process_msa(msa_data):
    """
    Process Multiple Sequence Alignment (MSA) data.
    
    This function converts the MSA data into a tensor representation,
    which is crucial for capturing evolutionary information in AlphaFold 3.
    MSAs provide valuable information about conserved regions and
    co-evolving residues, which helps in predicting protein structure.
    """
    msa_file = StringIO(msa_data)
    sequences = list(SeqIO.parse(msa_file, "fasta"))
    msa_tensor = torch.zeros(len(sequences), len(sequences[0]))
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq.seq):
            msa_tensor[i, j] = ord(aa) - ord('A')
    return msa_tensor

class BasicCNNModel(nn.Module):
    """
    A basic CNN model for comparison.
    
    This model serves as a baseline to demonstrate the improvements
    achieved by more advanced architectures like AlphaFold 3.
    """
    def __init__(self, seq_length, num_residues, embedding_dim):
        super(BasicCNNModel, self).__init__()
        self.embedding = nn.Embedding(26, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * seq_length, 128)
        self.fc2 = nn.Linear(128, num_residues * 3)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.embedding(x.long())
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.view(x.size(0), -1, 3)

class SimpleAttentionModel(nn.Module):
    """
    A simple attention-based model for comparison.
    
    This model incorporates basic attention mechanisms, showing a step
    towards more sophisticated architectures like AlphaFold 3.
    """
    def __init__(self, seq_length, num_residues, embedding_dim):
        super(SimpleAttentionModel, self).__init__()
        self.embedding = nn.Embedding(26, embedding_dim)
        self.attention = nn.MultiheadAttention(embedding_dim, 4)
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64 * seq_length, num_residues * 3)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.embedding(x.long())
        x = x.transpose(0, 1)
        x, _ = self.attention(x, x, x)
        x = x.transpose(0, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = x.view(x.size(0), -1)
        x = self.fc3(x)
        return x.view(x.size(0), -1, 3)

class Pairformer(nn.Module):
    """
    Implementation of the Pairformer module, a key component of AlphaFold 3.
    
    The Pairformer replaces the Evoformer from AlphaFold 2, providing a more
    efficient way to process pairwise representations. This module is crucial
    for capturing long-range dependencies in protein structures.
    """
    def __init__(self, dim):
        super(Pairformer, self).__init__()
        self.attention = nn.MultiheadAttention(dim, 4)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attention(x, x, x)[0]
        x = self.norm1(x)
        x = x + self.mlp(x)
        x = self.norm2(x)
        return x

class EnhancedDiffusionModule(nn.Module):
    """
    Implementation of the Enhanced Diffusion Module, inspired by AlphaFold 3.
    
    This module incorporates the key innovations of AlphaFold 3:
    1. Diffusion-based generation of protein structures
    2. Pairformer for efficient processing of pairwise representations
    3. Confidence prediction for each residue
    
    The diffusion approach allows for more accurate and diverse structure predictions,
    while the confidence prediction helps in assessing the reliability of the model's output.
    """
    def __init__(self, seq_length, num_residues, embedding_dim, num_pairformer_layers=3):
        super(EnhancedDiffusionModule, self).__init__()
        self.seq_length = seq_length
        self.num_residues = num_residues
        self.embedding_dim = embedding_dim
        
        # MSA embedding to capture evolutionary information
        self.msa_embedding = nn.Embedding(26, embedding_dim)
        
        # Pairformer layers for processing pairwise representations
        self.pairformers = nn.ModuleList([Pairformer(embedding_dim) for _ in range(num_pairformer_layers)])
        
        # Encoder to process input coordinates and MSA information
        self.encoder = nn.Sequential(
            nn.Linear(num_residues * 3 + seq_length * embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim * num_residues)
        )
        
        # Decoder to generate 3D coordinates
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * num_residues, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, num_residues * 3)
        )
        
        # Confidence predictor to assess the reliability of predictions
        self.confidence_predictor = nn.Sequential(
            nn.Linear(embedding_dim * num_residues, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, num_residues)
        )
        
    def forward(self, x, msa, t):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        # Process MSA information
        msa_embed = self.msa_embedding(msa.long()).mean(dim=0)
        combined = torch.cat([x_flat, msa_embed.flatten().unsqueeze(0).expand(batch_size, -1)], dim=1)
        
        # Encode input
        h = self.encoder(combined)
        
        # Apply Pairformer layers
        h = h.view(batch_size, self.num_residues, self.embedding_dim)
        for pairformer in self.pairformers:
            h = pairformer(h)
        h = h.view(batch_size, -1)
        
        # Add noise for diffusion process
        noise = torch.randn_like(h) * torch.sqrt(t.view(-1, 1))
        h_noisy = h + noise
        
        # Decode to generate 3D coordinates
        x_pred = self.decoder(h_noisy)
        x_pred = x_pred.view(batch_size, self.num_residues, 3)
        
        # Predict confidence scores
        confidence = self.confidence_predictor(h_noisy)
        confidence = torch.sigmoid(confidence)
        
        return x_pred, confidence

def diffusion_loss(model, x_0, msa, num_timesteps=1000):
    """
    Compute the diffusion loss for training the AlphaFold 3 inspired model.
    
    This loss function implements the diffusion process, which allows the model
    to learn a gradual denoising of protein structures. It also incorporates
    a confidence loss to improve the model's uncertainty estimation.
    """
    batch_size = x_0.shape[0]
    t = torch.randint(0, num_timesteps, (batch_size,), device=x_0.device).float() / num_timesteps
    
    # Add noise to the input coordinates
    noise = torch.randn_like(x_0)
    x_noisy = x_0 + noise * torch.sqrt(t.view(-1, 1, 1))
    
    # Generate predictions and confidence scores
    x_pred, confidence = model(x_noisy, msa, t)
    
    # Compute reconstruction loss
    reconstruction_loss = F.mse_loss(x_pred, x_0)
    
    # Compute confidence loss
    confidence_loss = F.mse_loss(confidence, torch.exp(-reconstruction_loss.detach()).mean().expand_as(confidence))
    
    # Combine losses
    total_loss = reconstruction_loss + 0.1 * confidence_loss
    return total_loss

def load_protein_structure(pdb_data):
    """
    Load protein structure from PDB data.
    
    This function parses PDB data to extract 3D coordinates of Cα atoms,
    which are used as the ground truth for training and evaluation.
    """
    parser = PDBParser()
    structure = parser.get_structure("protein", StringIO(pdb_data))
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ':  # Check if it's a standard amino acid
                    coords.append(residue['CA'].get_coord())
    return np.array(coords)

def coords_to_rdkit_mol(coords, residue_names):
    """
    Convert coordinates to RDKit molecule object.
    
    This function is not directly related to AlphaFold 3 but can be useful
    for visualizing or further processing the predicted structures.
    """
    mol = Chem.RWMol()
    for i, (coord, res_name) in enumerate(zip(coords, residue_names)):
        atom = Chem.Atom(6)  # Carbon atom as placeholder
        atom.SetProp("name", f"{res_name}_CA")
        atom_idx = mol.AddAtom(atom)
        mol.GetConformer().SetAtomPosition(atom_idx, coord)
    return mol

def train_and_evaluate(model, x_0, msa, num_epochs=1000):
    """
    Train and evaluate the given model.
    
    This function implements the training loop for all models, including
    the AlphaFold 3 inspired model. It uses different loss functions
    depending on the model type, showcasing the unique aspects of
    training a diffusion-based model compared to traditional approaches.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    losses = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        if isinstance(model, EnhancedDiffusionModule):
            # For the AlphaFold 3 inspired model, use the diffusion loss
            loss = diffusion_loss(model, x_0, msa)
        else:
            # For other models, use a simple MSE loss
            pred = model(msa[0].unsqueeze(0))
            loss = F.mse_loss(pred, x_0)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    return losses


if __name__ == "__main__":
    # Load the hard-coded protein structure
    coords = load_protein_structure(PDB_DATA)
    num_residues = len(coords)
    
    # Process the hard-coded MSA
    msa = process_msa(MSA_DATA)
    seq_length = msa.shape[1]
    
    # Convert coordinates to tensor
    x_0 = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    print(f"Number of residues: {num_residues}")
    print(f"MSA shape: {msa.shape}")
    print(f"Coordinates shape: {x_0.shape}")
    
    # Create and train models
    embedding_dim = 128
    cnn_model = BasicCNNModel(seq_length, num_residues, embedding_dim)
    attention_model = SimpleAttentionModel(seq_length, num_residues, embedding_dim)
    alphafold_model = EnhancedDiffusionModule(seq_length, num_residues, embedding_dim)
    
    print("Training CNN Model:")
    cnn_losses = train_and_evaluate(cnn_model, x_0, msa)
    
    print("\nTraining Simple Attention Model:")
    attention_losses = train_and_evaluate(attention_model, x_0, msa)
    
    print("\nTraining AlphaFold-inspired Model:")
    alphafold_losses = train_and_evaluate(alphafold_model, x_0, msa)
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(cnn_losses, label='CNN Model')
    plt.plot(attention_losses, label='Simple Attention Model')
    plt.plot(alphafold_losses, label='AlphaFold-inspired Model')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves for Different Models')
    plt.legend()
    plt.yscale('log')
    plt.savefig('learning_curves.png')
    plt.close()
    
    print("\nLearning curves have been saved to 'learning_curves.png'")
    
    # Generate predictions
    with torch.no_grad():
        cnn_pred = cnn_model(msa[0].unsqueeze(0))
        attention_pred = attention_model(msa[0].unsqueeze(0))
        # For the AlphaFold-inspired model, we need to provide noisy input and time step
        x_noisy = torch.randn_like(x_0)
        t = torch.ones(1, device=x_0.device)
        alphafold_pred, confidence = alphafold_model(x_noisy, msa, t)
    
    # Calculate final MSE for each model
    cnn_mse = F.mse_loss(cnn_pred, x_0).item()
    attention_mse = F.mse_loss(attention_pred, x_0).item()
    alphafold_mse = F.mse_loss(alphafold_pred, x_0).item()
    
    print(f"\nFinal MSE:")
    print(f"CNN Model: {cnn_mse:.4f}")
    print(f"Simple Attention Model: {attention_mse:.4f}")
    print(f"AlphaFold-inspired Model: {alphafold_mse:.4f}")
    
    print("\nNote: The AlphaFold-inspired model also provides confidence scores for its predictions.")