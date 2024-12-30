import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from openai import OpenAI
import json

# Initialize OpenAI client with API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class TagResponse:
    """JSON response format for tag generation"""
    tags: List[str]

@dataclass
class SummaryResponse:
    """JSON response format for class summary generation"""
    summary: str

@dataclass
class ClassificationResponse:
    """JSON response format for classification"""
    class_name: str
    confidence: float
    is_out_of_domain: bool = False

class GaussianClassModel:
    """Implements the External Continual Learner (ECL) using Gaussian distributions"""
    
    def __init__(self, embedding_dim: int = 1536):
        self.means = {}  # Class means
        self.shared_covariance = np.eye(embedding_dim) * 0.1  # Initialize with small variance
        self.class_count = 0
        self.min_covar_eigenval = 1e-6  # Minimum eigenvalue for numerical stability
        
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text string using OpenAI's embedding model"""
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return np.array(response.data[0].embedding)
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for multiple texts"""
        if not texts:
            return np.array([])
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return np.array([data.embedding for data in response.data])
        
    def _stabilize_covariance(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Ensure covariance matrix is numerically stable"""
        # Add small constant to diagonal for numerical stability
        cov_matrix += np.eye(cov_matrix.shape[0]) * self.min_covar_eigenval
        
        # Ensure symmetry
        cov_matrix = (cov_matrix + cov_matrix.T) / 2
        
        # Ensure positive definiteness through eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        eigvals = np.maximum(eigvals, self.min_covar_eigenval)
        cov_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        return cov_matrix
    
    def update_class_statistics(self, class_name: str, tags: List[str]):
        """Update Gaussian statistics for a class"""
        embeddings = self._get_embeddings(tags)
        if len(embeddings) == 0:
            return
            
        # Update mean for the class
        if class_name not in self.means:
            self.means[class_name] = np.mean(embeddings, axis=0)
            self.class_count += 1
        else:
            # Incremental mean update
            old_mean = self.means[class_name]
            n = len(embeddings)
            self.means[class_name] = (old_mean + np.mean(embeddings, axis=0)) / 2
        
        # Update shared covariance matrix
        diff = embeddings - self.means[class_name]
        class_cov = (diff.T @ diff) / max(len(tags), 1)
        
        # Update shared covariance with stability check
        if self.class_count > 1:
            self.shared_covariance = ((self.class_count - 1) * self.shared_covariance + class_cov) / self.class_count
        else:
            self.shared_covariance = class_cov
            
        self.shared_covariance = self._stabilize_covariance(self.shared_covariance)
    
    def get_top_k_classes(self, query_tags: List[str], k: int) -> tuple[List[str], List[float]]:
        """Get top k most similar classes using Mahalanobis distance"""
        if not query_tags:
            return [], []
            
        query_embeddings = self._get_embeddings(query_tags)
        if len(query_embeddings) == 0:
            return [], []
            
        query_mean = np.mean(query_embeddings, axis=0)
        
        # Calculate Mahalanobis distance to each class
        distances = {}
        inv_cov = np.linalg.inv(self.shared_covariance)
        
        for class_name, class_mean in self.means.items():
            diff = query_mean - class_mean
            dist = np.sqrt(max(0, diff.T @ inv_cov @ diff))  # Ensure non-negative
            distances[class_name] = dist.item()
        
        # Convert distances to similarity scores (inverse of distance)
        similarities = {cls: 1.0 / (dist + 1e-6) for cls, dist in distances.items()}
        
        # Normalize similarities to [0, 1]
        max_sim = max(similarities.values()) + 1e-6
        similarities = {cls: sim/max_sim for cls, sim in similarities.items()}
        
        # Sort by similarity (highest first)
        sorted_classes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        classes, scores = zip(*sorted_classes[:k])
        
        return list(classes), list(scores)

class InCA:
    """Implementation of In-context Continual Learning Assisted by an ECL"""
    
    def __init__(self, embedding_dim: int = 1536):
        self.ecl = GaussianClassModel(embedding_dim)
        self.class_summaries = {}
        self.confidence_threshold = 0.3
        
    def _calculate_semantic_similarity(self, query: str, class_summary: str) -> float:
        """Calculate semantic similarity between query and class summary"""
        if not class_summary:
            return 0.0
            
        query_emb = self.ecl._get_embedding(query)
        summary_emb = self.ecl._get_embedding(class_summary)
        
        similarity = np.dot(query_emb, summary_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(summary_emb)
        )
        
        return max(0, (similarity + 1) / 2)  # Normalize to [0, 1]
        
    def _generate_tags(self, query: str, examples: List[Dict[str, List[str]]]) -> List[str]:
        """Generate semantic tags using GPT-4o-mini with JSON mode"""
        prompt = {
            "role": "system",
            "content": """You are a tag generator that creates semantic tags for text input. 
                         Generate 5-10 relevant tags that capture the key concepts and intent.
                         Respond in JSON format with a 'tags' array."""
        }
        
        query_request = {
            "query": query,
            "examples": examples
        }
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                prompt,
                {"role": "user", "content": json.dumps(query_request)}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result.get("tags", [])
        except:
            return []
            
    def _generate_class_summary(self, class_name: str, examples: List[str]) -> str:
        """Generate class summary using GPT-4o-mini with JSON mode"""
        prompt = {
            "role": "system",
            "content": """You are a class summarizer that creates concise descriptions.
                         Create a clear, specific summary of the class based on examples.
                         Respond in JSON format with a 'summary' field."""
        }
        
        summary_request = {
            "class_name": class_name,
            "examples": examples
        }
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                prompt,
                {"role": "user", "content": json.dumps(summary_request)}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result.get("summary", "")
        except:
            return ""
            
    def learn_class(self, class_name: str, examples: List[str], tag_examples: List[Dict[str, List[str]]]):
        """Learn a new class incrementally"""
        summary = self._generate_class_summary(class_name, examples)
        self.class_summaries[class_name] = summary
        
        all_tags = []
        for example in examples:
            tags = self._generate_tags(example, tag_examples)
            all_tags.extend(tags)
        
        self.ecl.update_class_statistics(class_name, all_tags)
        
    def predict(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Predict class for new query with improved confidence and out-of-domain detection"""
        tags = self._generate_tags(query, [])
        top_classes, similarities = self.ecl.get_top_k_classes(tags, k)
        
        if not top_classes:
            return {
                "class_name": "unknown",
                "confidence": 0.0,
                "is_out_of_domain": True,
                "candidate_classes": [],
                "statistical_confidence": 0.0,
                "semantic_confidence": 0.0
            }
        
        # Calculate statistical confidence
        statistical_confidence = similarities[0] if similarities else 0.0
        
        # Calculate semantic confidence
        max_class = top_classes[0]
        class_summary = self.class_summaries.get(max_class, "")
        semantic_confidence = self._calculate_semantic_similarity(query, class_summary)
        
        # Combined confidence score
        final_confidence = 0.6 * statistical_confidence + 0.4 * semantic_confidence
        
        # Out-of-domain detection
        is_out_of_domain = (semantic_confidence < self.confidence_threshold and 
                           statistical_confidence < self.confidence_threshold)
        
        return {
            "class_name": max_class,
            "confidence": final_confidence,
            "is_out_of_domain": is_out_of_domain,
            "candidate_classes": top_classes,
            "semantic_confidence": semantic_confidence,
            "statistical_confidence": statistical_confidence
        }

def main():
    """Example usage demonstrating concept drift and zero-shot extension"""
    # Initialize InCA
    inca = InCA()
    
    # Example tag templates
    tag_examples = [
        {
            "query": "Getting errors while saving to floppy disk",
            "tags": ["storage", "error", "data", "hardware", "save", "disk_issue"]
        },
        {
            "query": "Cloud backup not syncing",
            "tags": ["storage", "cloud", "sync", "backup", "data", "connectivity"]
        }
    ]
    
    # Phase 1: Legacy Technology
    print("\n=== Phase 1: Training with Legacy Technology ===")
    legacy_classes = {
        "storage_issue": [
            "Floppy disk not reading",
            "CD-ROM drive making noise",
            "Can't format my diskette",
            "Zip drive not recognized"
        ],
        "display_problem": [
            "Monitor showing blue screen",
            "CRT display flickering",
            "Screen resolution too low",
            "Monitor colors look wrong"
        ]
    }
    
    for class_name, examples in legacy_classes.items():
        print(f"\nLearning legacy class: {class_name}")
        inca.learn_class(class_name, examples, tag_examples)
    
    # Test cases for each phase
    for phase, queries in [
        ("Legacy Technology", [
            "My floppy drive isn't working",
            "CRT monitor showing ghost images",
            "Tape backup failed"
        ]),
        ("Modern Technology (Concept Drift)", [
            "Cloud storage not syncing with my device",
            "Can't access my Google Drive files",
            "4K display has dead pixels",
            "OLED screen showing burn-in",
            "NVMe SSD not showing up in BIOS"
        ])
    ]:
        print(f"\n=== Testing with {phase} ===")
        for query in queries:
            result = inca.predict(query)
            print(f"\nQuery: {query}")
            print(f"Predicted class: {result['class_name']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Out of domain: {result['is_out_of_domain']}")
            print(f"Statistical confidence: {result['statistical_confidence']:.2f}")
            print(f"Semantic confidence: {result['semantic_confidence']:.2f}")
    
    # Phase 3: Zero-shot Extension
    print("\n=== Phase 3: Zero-shot Class Extension ===")
    new_class = {
        "connectivity_issue": [
            "WiFi keeps disconnecting",
            "Network connection unstable"
        ]
    }
    
    for class_name, examples in new_class.items():
        print(f"\nLearning new class with minimal examples: {class_name}")
        inca.learn_class(class_name, examples, tag_examples)
    
    # Test zero-shot queries
    zero_shot_queries = [
        "Bluetooth not pairing with my device",
        "5G signal drops in my area",
        "VPN connection keeps timing out",
        "Ethernet port not detecting cable",
        "DNS server not responding"
    ]
    
    print("\nTesting zero-shot generalization:")
    for query in zero_shot_queries:
        result = inca.predict(query)
        print(f"\nQuery: {query}")
        print(f"Predicted class: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Out of domain: {result['is_out_of_domain']}")
        print(f"Statistical confidence: {result['statistical_confidence']:.2f}")
        print(f"Semantic confidence: {result['semantic_confidence']:.2f}")

if __name__ == "__main__":
    main()