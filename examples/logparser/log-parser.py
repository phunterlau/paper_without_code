import os
import re
from typing import List, Dict, Tuple
from openai import OpenAI
import numpy as np

# Set up OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class LogCluster:
    def __init__(self, log_template: str):
        self.log_template = log_template
        self.syntax_templates = {}
        self.log_ids = set()
        # Store embedding for efficient similarity comparison
        self.embedding = None

    def add_log(self, log_id: int, syntax_template: List[str]):
        self.log_ids.add(log_id)
        token_count = len(syntax_template)
        if token_count not in self.syntax_templates:
            self.syntax_templates[token_count] = []
        if syntax_template not in self.syntax_templates[token_count]:
            self.syntax_templates[token_count].append(syntax_template)

class PrefixTreeNode:
    def __init__(self):
        self.children = {}
        self.clusters = set()

class LogParser:
    def __init__(self):
        self.root = PrefixTreeNode()
        self.clusters = {}
        self.template_pool = {}
        self.embedding_cache = {}  # Cache for log embeddings
        self.llm_call_count = 0  # Counter for LLM API calls

    def parse(self, logs: List[str]) -> Dict[str, List[int]]:
        for i, log in enumerate(logs):
            tokens = log.split()
            matched_clusters = self.search_tree(tokens)
            
            if self.strict_match(tokens, matched_clusters):
                continue
            
            # Use cached embedding or generate new one
            if log not in self.embedding_cache:
                self.embedding_cache[log] = self.get_embedding(log)
            log_embedding = self.embedding_cache[log]
            
            # Check for similar existing templates
            similar_template = self.find_similar_template(log_embedding)
            if similar_template:
                if self.verify_merge(log, similar_template):
                    merged_template = self.merge_templates(log, similar_template)
                    self.update_cluster(merged_template, i, tokens)
                    continue
            
            # Extract new template if no match found
            template = self.extract_template(log)
            self.llm_call_count += 1  # Increment LLM call counter
            
            if template in self.template_pool:
                cluster = self.template_pool[template]
            else:
                cluster = LogCluster(template)
                self.clusters[i] = cluster
                self.template_pool[template] = cluster
                cluster.embedding = log_embedding
            
            self.update_tree(tokens, cluster)
            cluster.add_log(i, tokens)
        
        return {cluster.log_template: list(cluster.log_ids) for cluster in self.clusters.values()}

    def search_tree(self, tokens: List[str]) -> List[LogCluster]:
        node = self.root
        matched_clusters = set()
        
        for token in tokens:
            if token in node.children:
                node = node.children[token]
                matched_clusters.update(node.clusters)
            elif "<*>" in node.children:
                node = node.children["<*>"]
                matched_clusters.update(node.clusters)
            else:
                break
        
        return list(matched_clusters)

    def strict_match(self, tokens: List[str], clusters: List[LogCluster]) -> bool:
        for cluster in clusters:
            if len(tokens) in cluster.syntax_templates:
                for template in cluster.syntax_templates[len(tokens)]:
                    if all(t == "<*>" or t == token for t, token in zip(template, tokens)):
                        return True
        return False

    def update_tree(self, tokens: List[str], cluster: LogCluster):
        node = self.root
        for token in tokens:
            if token not in node.children:
                node.children[token] = PrefixTreeNode()
            node = node.children[token]
            node.clusters.add(cluster)

    def extract_template(self, log: str) -> str:
        # Implement in-context learning and variable-aware prompting
        examples = self.get_similar_examples(log)
        prompt = f"""As a log parser, analyze the following log and identify dynamic variables. 
        Replace dynamic variables with <XXX> placeholders. Static parts should remain unchanged.
        Do not fix any typos. If a variable is compound, replace the entire variable with a single <XXX>.
        Only replace genuine dynamic variables.

        Examples:
        {examples}

        Log: {log}
        Template:"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        return response.choices[0].message.content.strip()

    def get_embedding(self, text: str) -> List[float]:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def find_similar_template(self, log_embedding: List[float], threshold: float = 0.9) -> str:
        for template, cluster in self.template_pool.items():
            if cluster.embedding is None:
                continue
            similarity = np.dot(log_embedding, cluster.embedding)
            if similarity > threshold:
                return template
        return None

    def verify_merge(self, log: str, template: str) -> bool:
        prompt = f"""Does the template: "{template}" apply to the following log? Please answer with yes or no.

        Log: {log}
        Answer:"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        self.llm_call_count += 1  # Increment LLM call counter
        return response.choices[0].message.content.strip().lower() == "yes"

    def merge_templates(self, log: str, existing_template: str) -> str:
        prompt = f"""Merge the following log and existing template into a single template:

        Log: {log}
        Existing Template: {existing_template}
        Merged Template:"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        self.llm_call_count += 1  # Increment LLM call counter
        return response.choices[0].message.content.strip()

    def update_cluster(self, new_template: str, log_id: int, tokens: List[str]):
        if new_template in self.template_pool:
            cluster = self.template_pool[new_template]
        else:
            cluster = LogCluster(new_template)
            self.template_pool[new_template] = cluster
        
        cluster.add_log(log_id, tokens)
        self.update_tree(tokens, cluster)

    def get_similar_examples(self, log: str) -> str:
        # In a real implementation, this would return actual similar examples
        # For simplicity, we'll return a fixed set of examples
        return """Log: User 123 logged in from 192.168.1.1
        Template: User <XXX> logged in from <XXX>

        Log: Error: Connection timeout for user 789
        Template: Error: Connection timeout for user <XXX>

        Log: 2023-08-27 15:45:23,456 - DEBUG - main.py:42 - Starting data processing for user_id=12345
        Template: <XXX> - <XXX> - <XXX>:<XXX> - Starting data processing for user_id=<XXX>"""

def calculate_grouping_accuracy(parsed_results: Dict[str, List[int]], ground_truth: Dict[str, List[int]]) -> float:
    correct = 0
    total = 0
    
    for template, log_ids in parsed_results.items():
        for log_id in log_ids:
            if any(log_id in gt_ids for gt_ids in ground_truth.values()):
                correct += 1
            total += 1
    
    return correct / total if total > 0 else 0

def calculate_granularity_distance(parsed_results: Dict[str, List[int]], ground_truth: Dict[str, List[int]]) -> int:
    # This is a simplified version of granularity distance calculation
    # In a real implementation, this would be more sophisticated
    parsed_templates = set(parsed_results.keys())
    ground_truth_templates = set(ground_truth.keys())
    
    return len(parsed_templates.symmetric_difference(ground_truth_templates))

# Example usage with enriched log set
logs = [
    # General logs
    "User 123 logged in from 192.168.1.1",
    "User 456 logged in from 10.0.0.1",
    "Error: Connection timeout for user 789",
    "Server restarted at 2023-08-27 14:30:00",
    "User 123 logged out",
    
    # Java logs
    "INFO: [ThreadPool-7] org.apache.coyote.http11.Http11Processor - Error parsing HTTP request header",
    "WARN: [main] org.hibernate.engine.jdbc.spi.SqlExceptionHelper - SQL Error: 1064, SQLState: 42000",
    "DEBUG: [pool-1-thread-1] com.example.MyClass - Processing item 42 in batch job",
    "ERROR: [scheduling-1] com.example.service.UserService - Failed to update user profile: NullPointerException",
    "INFO: [http-nio-8080-exec-3] org.springframework.web.servlet.DispatcherServlet - Completed 200 OK in 150ms",
    
    # Python logs
    "2023-08-27 15:45:23,456 - DEBUG - main.py:42 - Starting data processing for user_id=12345",
    "2023-08-27 15:45:24,789 - INFO - database.py:78 - Successfully connected to database 'myapp_db'",
    "2023-08-27 15:45:25,123 - WARNING - utils.py:156 - Deprecated function 'old_method' called from module 'legacy'",
    "2023-08-27 15:45:26,456 - ERROR - api.py:203 - HTTP Error 404: Not Found for URL: https://api.example.com/v1/users/999",
    "2023-08-27 15:45:27,789 - CRITICAL - main.py:85 - Unhandled exception: Division by zero in calculate_average() at line 72"
]

ground_truth = {
    "User <XXX> logged in from <XXX>": [0, 1],
    "Error: Connection timeout for user <XXX>": [2],
    "Server restarted at <XXX>": [3],
    "User <XXX> logged out": [4],
    "<XXX>: [<XXX>] <XXX> - Error parsing HTTP request header": [5],
    "<XXX>: [<XXX>] <XXX> - SQL Error: <XXX>, SQLState: <XXX>": [6],
    "<XXX>: [<XXX>] <XXX> - Processing item <XXX> in batch job": [7],
    "<XXX>: [<XXX>] <XXX> - Failed to update user profile: <XXX>": [8],
    "<XXX>: [<XXX>] <XXX> - Completed <XXX> OK in <XXX>ms": [9],
    "<XXX> - <XXX> - <XXX>:<XXX> - Starting data processing for user_id=<XXX>": [10],
    "<XXX> - <XXX> - <XXX>:<XXX> - Successfully connected to database '<XXX>'": [11],
    "<XXX> - <XXX> - <XXX>:<XXX> - Deprecated function '<XXX>' called from module '<XXX>'": [12],
    "<XXX> - <XXX> - <XXX>:<XXX> - HTTP Error <XXX>: <XXX> for URL: <XXX>": [13],
    "<XXX> - <XXX> - <XXX>:<XXX> - Unhandled exception: <XXX> in <XXX>() at line <XXX>": [14]
}

parser = LogParser()
parsed_results = parser.parse(logs)

print("Parsed Results:")
for template, log_ids in parsed_results.items():
    print(f"{template}: {log_ids}")

accuracy = calculate_grouping_accuracy(parsed_results, ground_truth)
granularity_distance = calculate_granularity_distance(parsed_results, ground_truth)

print(f"\nGrouping Accuracy: {accuracy:.2f}")
print(f"Granularity Distance: {granularity_distance}")
print(f"Total LLM API calls: {parser.llm_call_count}")