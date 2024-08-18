import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

# Layer Normalization: Normalizes the inputs to each layer, which helps with training stability
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# Masked Multi-Head Attention: Core component of the Transformer, allows the model to attend to different parts of the input
class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MaskedMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # 1. Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        # 2. Apply attention on all the projected vectors in batch
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        x = torch.matmul(p_attn, value)
        # 3. "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.output_linear(x)

# Position-wise Feed-Forward Network: Applies two linear transformations with a ReLU activation in between
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w_2(F.relu(self.w_1(x)))

# Transformer Block: Combines Multi-Head Attention and Feed-Forward layers
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MaskedMultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm1(x)
        x = x + self.dropout1(self.attention(x2, x2, x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x

# GPT Model: The main model architecture
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=12, num_heads=12, d_ff=3072, dropout=0.1):
        super(GPT, self).__init__()
        self.d_model = d_model
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional embeddings
        self.pos_embedding = nn.Embedding(1024, d_model)  # Max sequence length of 1024
        # Transformer layers
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)
        # Output layer
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, mask=None):
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0).expand(b, t)
        x = self.embedding(x) + self.pos_embedding(pos)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x, self.lm_head(x)

# GPT for Sequence Classification: Adapts the GPT model for classification tasks
class GPTForSequenceClassification(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(GPTForSequenceClassification, self).__init__()
        self.transformer = pretrained_model
        self.classifier = nn.Linear(self.transformer.d_model, num_classes)

    def forward(self, x, mask=None):
        transformer_output, lm_logits = self.transformer(x, mask)
        seq_output = transformer_output[:, -1, :]  # Use the last token's representation for classification
        logits = self.classifier(seq_output)
        return logits, lm_logits

# Simple Tokenizer: Converts text to token ids and vice versa
class SimpleTokenizer:
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>"}
        self.word_count = 2

    def encode(self, text):
        return torch.tensor([self.word_to_idx.get(word, 1) for word in text.split()])

    def decode(self, tokens):
        return " ".join([self.idx_to_word.get(t.item(), "<UNK>") for t in tokens])

    def add_word(self, word):
        if word not in self.word_to_idx:
            self.word_to_idx[word] = self.word_count
            self.idx_to_word[self.word_count] = word
            self.word_count += 1

# Create attention mask for self-attention
def create_masks(input_ids):
    seq_length = input_ids.size(1)
    subsequent_mask = torch.triu(torch.ones((1, seq_length, seq_length)), diagonal=1).bool()
    return subsequent_mask.to(input_ids.device)

# Pre-training function: Trains the model on the language modeling task
def pretrain(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = create_masks(input_ids).to(device)
        labels = input_ids[:, 1:].contiguous()  # Shift right for next-token prediction

        optimizer.zero_grad()
        _, lm_logits = model(input_ids, attention_mask)
        loss = criterion(lm_logits[:, :-1, :].contiguous().view(-1, lm_logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)

# Fine-tuning function: Adapts the pre-trained model for specific tasks
def fine_tune(model, data_loader, optimizer, criterion, device, lm_coef=0.5):
    model.train()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = create_masks(input_ids).to(device)
        labels = batch['labels'].to(device)
        lm_labels = input_ids[:, 1:].contiguous()

        optimizer.zero_grad()
        logits, lm_logits = model(input_ids, attention_mask)
        classification_loss = criterion(logits, labels)
        lm_loss = F.cross_entropy(lm_logits[:, :-1, :].contiguous().view(-1, lm_logits.size(-1)), lm_labels.view(-1))
        loss = classification_loss + lm_coef * lm_loss  # Combine task-specific and language modeling losses
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)

# Example data for different tasks
pretrain_data = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question.",
    "All that glitters is not gold.",
    "Where there's a will, there's a way."
]

fine_tune_data = [
    ("This movie is great!", 1),  # Positive sentiment
    ("I didn't like the book at all.", 0),  # Negative sentiment
    ("The restaurant was okay.", 1),  # Positive sentiment
    ("Terrible service and bad food.", 0),  # Negative sentiment
    ("I love this product!", 1)  # Positive sentiment
]

entailment_data = [
    ("The cat is on the mat.", "There is a cat.", 1),  # Entailment
    ("The dog is barking.", "The animal is sleeping.", 0),  # Contradiction
    ("It's raining outside.", "The ground might be wet.", 1),  # Entailment
    ("The sun is shining.", "It's nighttime.", 0),  # Contradiction
    ("The car is red.", "The vehicle has a color.", 1)  # Entailment
]

# Prepare tokenizer and build vocabulary
vocab_size = 1000
tokenizer = SimpleTokenizer(vocab_size)

for text in pretrain_data:
    for word in text.split():
        tokenizer.add_word(word)

for text, _ in fine_tune_data:
    for word in text.split():
        tokenizer.add_word(word)

for premise, hypothesis, _ in entailment_data:
    for word in premise.split() + hypothesis.split():
        tokenizer.add_word(word)

# Dataset class to handle different types of data
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, task='pretrain', max_length=20):
        self.data = data
        self.tokenizer = tokenizer
        self.task = task
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.task == 'pretrain':
            text = self.data[idx]
            input_ids = self.tokenizer.encode(text)
            input_ids = input_ids[:self.max_length]  # Truncate if too long
            return {'input_ids': input_ids}
        elif self.task == 'classification':
            text, label = self.data[idx]
            input_ids = self.tokenizer.encode(text)
            input_ids = input_ids[:self.max_length]  # Truncate if too long
            return {'input_ids': input_ids, 'labels': torch.tensor(label)}
        elif self.task == 'entailment':
            premise, hypothesis, label = self.data[idx]
            input_ids = self.tokenizer.encode(premise + " " + hypothesis)
            input_ids = input_ids[:self.max_length]  # Truncate if too long
            return {'input_ids': input_ids, 'labels': torch.tensor(label)}

# Collate function to handle batching of variable-length sequences
def collate_fn(batch):
    max_len = max([len(item['input_ids']) for item in batch])
    
    input_ids = [F.pad(item['input_ids'], (0, max_len - len(item['input_ids'])), value=0) for item in batch]
    input_ids = torch.stack(input_ids)
    
    if 'labels' in batch[0]:
        labels = torch.stack([item['labels'] for item in batch])
        return {'input_ids': input_ids, 'labels': labels}
    else:
        return {'input_ids': input_ids}

# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare datasets
    pretrain_dataset = SimpleDataset(pretrain_data, tokenizer, 'pretrain')
    fine_tune_dataset = SimpleDataset(fine_tune_data, tokenizer, 'classification')
    entailment_dataset = SimpleDataset(entailment_data, tokenizer, 'entailment')

    # Create data loaders
    batch_size = 2
    pretrain_loader = torch.utils.data.DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    fine_tune_loader = torch.utils.data.DataLoader(fine_tune_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    entailment_loader = torch.utils.data.DataLoader(entailment_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Pre-training
    model = GPT(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding index
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 5
    for epoch in range(num_epochs):
        loss = pretrain(model, pretrain_loader, optimizer, criterion, device)
        print(f"Pre-training Epoch {epoch+1}, Loss: {loss:.4f}")

    # Fine-tuning for classification
    num_classes = 2  # Binary sentiment classification
    fine_tuned_model = GPTForSequenceClassification(model, num_classes).to(device)
    fine_tune_criterion = nn.CrossEntropyLoss()
    fine_tune_optimizer = torch.optim.Adam(fine_tuned_model.parameters(), lr=6.25e-5)

    num_epochs = 5
    for epoch in range(num_epochs):
        loss = fine_tune(fine_tuned_model, fine_tune_loader, fine_tune_optimizer, fine_tune_criterion, device)
        print(f"Fine-tuning (Classification) Epoch {epoch+1}, Loss: {loss:.4f}")

    # Example classification
    text = "This product is amazing!"
    input_ids = tokenizer.encode(text).unsque