import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.neighbors import NearestNeighbors

print("Importing necessary libraries...")

class MRLTransformer(nn.Module):
    def __init__(self, bert_model, nesting_dims, num_classes):
        super(MRLTransformer, self).__init__()
        self.bert = bert_model
        self.nesting_dims = nesting_dims
        self.num_classes = num_classes
        
        # Create nested classifiers for each dimension in nesting_dims
        self.classifiers = nn.ModuleList([
            nn.Linear(dim, num_classes) for dim in nesting_dims
        ])
        print(f"Initialized MRLTransformer with nesting dimensions: {nesting_dims}")
    
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Apply nested classifiers to get predictions at different granularities
        nested_logits = []
        for i, classifier in enumerate(self.classifiers):
            nested_output = pooled_output[:, :self.nesting_dims[i]]
            nested_logits.append(classifier(nested_output))
        
        return nested_logits, pooled_output

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Created dataset with {len(texts)} examples")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize and encode the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_mrl_transformer(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    
    model.to(device)
    print(f"Model moved to device: {device}")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Training phase:")
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            nested_logits, _ = model(input_ids, attention_mask)
            
            # Calculate loss for all nested dimensions
            loss = sum(criterion(logits, labels) for logits in nested_logits)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        
        print("\nValidation phase:")
        model.eval()
        val_loss = 0.0
        correct = [0] * len(model.nesting_dims)
        total = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                nested_logits, _ = model(input_ids, attention_mask)
                
                loss = sum(criterion(logits, labels) for logits in nested_logits)
                val_loss += loss.item()
                
                for i, logits in enumerate(nested_logits):
                    _, predicted = torch.max(logits, 1)
                    correct[i] += (predicted == labels).sum().item()
                total += labels.size(0)
                
                if batch_idx % 5 == 0:
                    print(f"  Batch {batch_idx+1}/{len(val_loader)}, Loss: {loss.item():.4f}")
        
        val_loss /= len(val_loader)
        accuracies = [100 * c / total for c in correct]
        
        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        for i, acc in enumerate(accuracies):
            print(f'Accuracy at dim {model.nesting_dims[i]}: {acc:.2f}%')
        print()

class ANNRetrieval:
    def __init__(self, embeddings, nesting_dims):
        self.embeddings = embeddings
        self.nesting_dims = nesting_dims
        self.ann_indices = {}
        
        print("Initializing ANN indices for each nesting dimension...")
        for dim in nesting_dims:
            self.ann_indices[dim] = NearestNeighbors(n_neighbors=10, algorithm='auto')
            self.ann_indices[dim].fit(embeddings[:, :dim])
        print("ANN indices initialized")
    
    def retrieve(self, query, dim):
        distances, indices = self.ann_indices[dim].kneighbors([query[:dim]])
        return distances[0], indices[0]

# Main execution
if __name__ == "__main__":
    print("Setting up device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\nInitializing BERT model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    print("BERT model and tokenizer initialized")

    nesting_dims = [8, 16, 32, 64, 128, 256, 512, 768]
    num_classes = 4

    print("\nCreating MRLTransformer model...")
    model = MRLTransformer(bert_model, nesting_dims, num_classes)
    print("MRLTransformer model created")

    print("\nPreparing dataset...")
    texts = [
        "This movie was absolutely fantastic, I loved every minute of it!",
        "The food at this restaurant was terrible, I wouldn't recommend it to anyone.",
        "I'm feeling quite neutral about this product, it's neither good nor bad.",
        "The customer service was outstanding, they went above and beyond to help me.",
        "This book was a disappointment, it didn't live up to the hype at all.",
        "The concert was amazing, the band's energy was infectious!",
        "I found this article to be quite informative and well-written.",
        "The hotel room was a bit small, but overall it was a pleasant stay.",
        "This software is incredibly buggy, it's causing me a lot of frustration.",
        "The scenery on this hike was breathtaking, I'd highly recommend it.",
    ]
    labels = [0, 3, 1, 0, 3, 0, 1, 1, 3, 0]  # 0: Very Positive, 1: Somewhat Positive, 2: Somewhat Negative, 3: Very Negative

    print("\nCreating datasets and dataloaders...")
    train_dataset = TextDataset(texts, labels, tokenizer, max_length=128)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=4)
    print("Datasets and dataloaders created")

    print("\nStarting training...")
    num_epochs = 5
    train_mrl_transformer(model, train_loader, val_loader, num_epochs, device)
    print("Training completed")

    print("\nGenerating embeddings for ANN retrieval...")
    embeddings = []
    model.eval()
    with torch.no_grad():
        for text in texts:
            encoding = tokenizer.encode_plus(text, add_special_tokens=True, max_length=128, 
                                             padding='max_length', truncation=True, return_tensors='pt')
            _, pooled_output = model(encoding['input_ids'].to(device), encoding['attention_mask'].to(device))
            embeddings.append(pooled_output.cpu().numpy()[0])
    embeddings = np.array(embeddings)
    print(f"Generated embeddings shape: {embeddings.shape}")

    print("\nInitializing ANN retrieval...")
    ann_retrieval = ANNRetrieval(embeddings, nesting_dims)

    print("\nPerforming inference and retrieval on test examples...")
    test_texts = [
        "This product exceeded my expectations in every way.",
        "The service was okay, but there's definitely room for improvement.",
    ]

    model.eval()
    with torch.no_grad():
        for test_text in test_texts:
            print(f"\nTest text: {test_text}")
            encoding = tokenizer.encode_plus(
                test_text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            nested_logits, query_embedding = model(encoding['input_ids'].to(device), encoding['attention_mask'].to(device))

            print("Classification results:")
            for i, logits in enumerate(nested_logits):
                probs = torch.softmax(logits, dim=1)
                print(f"Dimension {nesting_dims[i]}: {probs[0]}")
            print("Class probabilities: Very Positive, Somewhat Positive, Somewhat Negative, Very Negative")

            print("\nRetrieval results:")
            for dim in nesting_dims:
                distances, indices = ann_retrieval.retrieve(query_embedding.cpu().numpy()[0], dim)
                print(f"\nTop 3 results for dimension {dim}:")
                for d, i in zip(distances[:3], indices[:3]):
                    print(f"  Distance: {d:.4f}, Text: {texts[i][:50]}...")

    print("\nScript execution completed")