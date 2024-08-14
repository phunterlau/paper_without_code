import os
import dspy
import openai
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up DSPy with GPT-4
dspy.settings.configure(lm=dspy.OpenAI(model="gpt-4o-mini"))

# Sample corpus of documents
sample_corpus = [
    "The Industrial Revolution began in Britain in the late 18th century and quickly spread to other parts of Europe and North America. It marked a major turning point in history, with almost every aspect of daily life influenced in some way.",
    "One of the key features of the Industrial Revolution was the transition from manual labor and animal-based production to machine-based manufacturing. This led to increased productivity and efficiency in various industries.",
    "The textile industry was one of the first to be transformed by the Industrial Revolution. The invention of the spinning jenny and the power loom revolutionized the production of cloth, making it faster and cheaper.",
    "The development of steam power was crucial to the Industrial Revolution. Steam engines were used to power factories, locomotives, and ships, greatly improving transportation and manufacturing capabilities.",
    "Urbanization was a significant consequence of the Industrial Revolution. As factories were built, people moved from rural areas to cities in search of work, leading to the rapid growth of urban centers.",
    "The Industrial Revolution led to significant social changes. The growth of factories created a new working class, and labor movements emerged to fight for better working conditions and rights.",
    "Environmental impact was a major side effect of the Industrial Revolution. The increased use of coal and other fossil fuels led to air and water pollution in industrial areas.",
    "The Industrial Revolution sparked a series of technological advancements. Innovations in metallurgy and chemical manufacturing laid the groundwork for future technological progress.",
    "Education was transformed during the Industrial Revolution. The need for a skilled workforce led to the expansion of elementary education and the establishment of technical schools.",
    "The Industrial Revolution had a profound impact on agriculture. New farming techniques and technologies increased food production, supporting the growing urban population."
]

def retrieve_documents(query, num_docs=5):
    return np.random.choice(sample_corpus, num_docs, replace=False).tolist()

def cluster_documents(documents, num_clusters=3):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(documents)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans.labels_

class RAGDrafter(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_draft = dspy.ChainOfThought("question, context -> answer, rationale")

    def forward(self, question, context):
        return self.generate_draft(question=question, context=context)

class RAGVerifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.verify = dspy.ChainOfThought("question, answer, rationale -> numeric_score")

    def forward(self, question, answer, rationale):
        result = self.verify(question=question, answer=answer, rationale=rationale)
        try:
            # Extract numeric value from the string
            numeric_string = re.search(r'\d+', result.numeric_score).group()
            return float(numeric_string)
        except (ValueError, AttributeError):
            # If extraction or conversion fails, return a default low score
            print(f"Warning: Could not extract or convert score to float: {result.numeric_score}")
            return 0.0

def speculative_rag(question, num_drafts=3, num_docs_per_draft=2):
    documents = retrieve_documents(question)
    clusters = cluster_documents(documents)
    drafter = RAGDrafter()
    verifier = RAGVerifier()

    best_answer = None
    best_score = -float('inf')

    for _ in range(num_drafts):
        sampled_docs = []
        for _ in range(num_docs_per_draft):
            cluster = np.random.choice(len(set(clusters)))
            doc_indices = [i for i, c in enumerate(clusters) if c == cluster]
            sampled_docs.append(documents[np.random.choice(doc_indices)])

        context = " ".join(sampled_docs)
        draft_result = drafter(question, context)
        score = verifier(question, draft_result.answer, draft_result.rationale)

        if score > best_score:
            best_score = score
            best_answer = draft_result.answer

    return best_answer

if __name__ == "__main__":
    question = "What was the impact of the Industrial Revolution on urban development?"
    answer = speculative_rag(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")