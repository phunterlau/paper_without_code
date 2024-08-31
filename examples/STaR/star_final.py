import os
import openai
import dspy
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import random

# Set up OpenAI API
openai.api_key = os.environ["OPENAI_API_KEY"]

# Set up DSPy with OpenAI
openai_lm = dspy.OpenAI(model="gpt-4o-mini", api_key=openai.api_key)
dspy.settings.configure(lm=openai_lm)

class ImprovedChainOfThoughtModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # Generate rationale without the answer
        self.generate_rationale = dspy.ChainOfThought("question, answer_choices -> rationale")
        # Generate answer based on the rationale
        self.generate_answer = dspy.ChainOfThought("question, answer_choices, rationale -> exact_answer, confidence")

    def forward(self, question: str, answer_choices: List[str], num_samples: int = 3):
        """
        Generate multiple rationales and answers, then select the best one.
        This method leverages knowledge from previous iterations by using the fine-tuned model.
        """
        responses = []
        for _ in range(num_samples):
            # Generate rationale first, utilizing knowledge from previous iterations
            rationale = self.generate_rationale(question=question, answer_choices=str(answer_choices)).rationale.strip()
            # Then generate answer based on the rationale
            answer_response = self.generate_answer(
                question=question,
                answer_choices=str(answer_choices),
                rationale=rationale
            )
            answer = self._clean_answer(answer_response.exact_answer.strip())
            confidence = self._extract_confidence(answer_response.confidence)
            responses.append((rationale, answer, confidence))
        
        # Use voting to select the final answer, improving robustness
        answers = [r[1] for r in responses]
        answer_counts = Counter(answers)
        most_common_answer = answer_counts.most_common(1)[0][0]
        
        # Calculate confidence based on agreement between samples
        agreement_confidence = answer_counts[most_common_answer] / num_samples
        
        # Select the rationale corresponding to the most common answer with highest confidence
        best_response = max(
            (r for r in responses if r[1] == most_common_answer),
            key=lambda r: r[2]
        )
        
        # Combine model-reported confidence with agreement confidence
        final_confidence = (best_response[2] + agreement_confidence) / 2
        
        return best_response[0], most_common_answer, final_confidence

    def _extract_confidence(self, confidence_str: str) -> float:
        """Extract confidence score from the model's output."""
        match = re.search(r'confidence:?\s*(\d+(?:\.\d+)?)', confidence_str, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return random.uniform(0.5, 0.7)  # Return a random value if no valid confidence is found

    def _clean_answer(self, answer: str) -> str:
        """Clean and standardize the answer string."""
        # Remove common prefixes
        cleaned = re.sub(r'^(short answer:|answer:|question:|rationale:)\s*', '', answer, flags=re.IGNORECASE)
        
        # Remove any text after a period or comma
        cleaned = cleaned.split('.')[0].split(',')[0]
        
        # Remove any text in parentheses
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        
        # Strip whitespace and convert to lowercase
        cleaned = cleaned.strip().lower()
        
        # Remove any non-alphanumeric characters at the start or end of the string
        cleaned = re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', cleaned)

        return cleaned
  
class ImprovedSTaR:
    def __init__(self, few_shot_examples: List[Tuple[str, str, str]]):
        self.few_shot_examples = few_shot_examples
        self.model = ImprovedChainOfThoughtModule()
        self.dataset = []
        self.client = openai.OpenAI()
        
    def generate_rationale(self, question: str, answer_choices: List[str]) -> Tuple[str, str, float]:
        """
        Generate a rationale and answer for a given question.
        This method uses the knowledge accumulated from previous iterations.
        """
        prompt = self._create_prompt(question, answer_choices)
        try:
            rationale, answer, confidence = self.model(question=prompt, answer_choices=answer_choices)
            
            # Check if the answer is relevant to the question
            if not self._is_answer_relevant(question, answer, answer_choices):
                print(f"Warning: Generated answer may not be relevant to the question.")
                return "", self._semantic_similarity_fallback(question, answer_choices), 0.0
            
        except Exception as e:
            print(f"Error generating rationale: {str(e)}")
            return "", self._semantic_similarity_fallback(question, answer_choices), 0.0
        
        return rationale, answer, confidence

    def _is_answer_relevant(self, question: str, answer: str, answer_choices: List[str]) -> bool:
        """Check if the generated answer is relevant to the question."""
        return any(choice.lower() in answer.lower() for choice in answer_choices)

    def rationalize(self, question: str, answer_choices: List[str], correct_answer: str) -> str:
        """
        Generate a rationale for a given correct answer.
        This method is used to improve the model's reasoning for incorrect answers.
        """
        prompt = self._create_rationalization_prompt(question, answer_choices, correct_answer)
        rationale, _, _ = self.model(question=prompt, answer_choices=answer_choices)
        return rationale

    def finetune(self):
        """
        Simulate fine-tuning by updating few-shot examples.
        This method accumulates knowledge from previous iterations.
        """
        self.few_shot_examples += self.dataset
        self.dataset = []

    def _create_prompt(self, question: str, answer_choices: List[str]) -> str:
        """
        Create a prompt for generating rationales and answers.
        This method incorporates few-shot examples from previous iterations.
        """
        prompt = ("You are an expert in answering complex questions. Provide a step-by-step rationale for your answer, "
                "considering all relevant aspects. Be thorough and logical. After your rationale, clearly state your "
                "final answer by copying EXACTLY one of the provided answer choices. Then, provide a confidence score "
                "between 0 and 1. Do not repeat the question or answer choices in your response.\n\n")
        
        relevant_examples = self._select_relevant_examples(question, 3)
        
        for example in relevant_examples:
            prompt += f"Q: {example[0]}\nAnswer Choices: {example[1]}\nA: {example[2]}\n\n"
        
        prompt += f"Q: {question}\nAnswer Choices: {answer_choices}\nA: Let's approach this step-by-step:\n\n"
        return prompt

    def _create_rationalization_prompt(self, question: str, answer_choices: List[str], correct_answer: str) -> str:
        """
        Create a prompt for rationalizing a given correct answer.
        This method is used to improve the model's reasoning for incorrect answers.
        """
        prompt = ("You are an expert in explaining complex concepts. Given the correct answer, provide a comprehensive "
                  "step-by-step rationale for why it is correct. Consider multiple angles and potential counterarguments.\n\n")
        
        relevant_examples = self._select_relevant_examples(question, 2)
        
        for example in relevant_examples:
            prompt += f"Q: {example[0]}\nAnswer Choices: {example[1]}\nCorrect Answer: {example[2].split('Therefore, the answer is')[1].strip()}\nA: {example[2]}\n\n"
        
        prompt += f"Q: {question}\nAnswer Choices: {answer_choices}\nCorrect Answer: {correct_answer}\nA: Let's break this down step-by-step:\n\n"
        return prompt

    def _get_embedding(self, text: str) -> List[float]:
        """Get the embedding for a given text using OpenAI's API."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def _select_relevant_examples(self, question: str, n: int) -> List[Tuple[str, str, str]]:
        """
        Select the most relevant few-shot examples for a given question.
        This method helps in transferring knowledge from previous iterations to new questions.
        """
        question_embedding = self._get_embedding(question)
        example_embeddings = [self._get_embedding(ex[0]) for ex in self.few_shot_examples]
        
        similarities = cosine_similarity([question_embedding], example_embeddings)[0]
        top_indices = np.argsort(similarities)[-n:][::-1]
        
        return [self.few_shot_examples[i] for i in top_indices]

    def _semantic_similarity_fallback(self, question: str, answer_choices: List[str]) -> str:
        """Fallback method to select an answer based on semantic similarity."""
        question_embedding = self._get_embedding(question)
        choice_embeddings = [self._get_embedding(choice) for choice in answer_choices]
        
        similarities = cosine_similarity([question_embedding], choice_embeddings)[0]
        return answer_choices[np.argmax(similarities)]

def evaluate_accuracy(star: ImprovedSTaR, test_set: List[Tuple[str, List[str], str]]) -> float:
    """
    Evaluate the accuracy of the model on a test set.
    This function helps in assessing how well the model generalizes to new questions.
    """
    correct = 0
    total = len(test_set)
    for question, answer_choices, correct_answer in test_set:
        rationale, predicted_answer, confidence = star.generate_rationale(question, answer_choices)
        if predicted_answer.lower().strip() == correct_answer.lower().strip():
            correct += 1
        print(f"Question: {question}")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Rationale: {rationale[:500]}...")  # Print first 500 characters of rationale
        print()
    return correct / total  

# Example usage
few_shot_examples = [
    ("What do people use to absorb extra ink from a fountain pen?", 
     ["shirt pocket", "calligrapher's hand", "inkwell", "desk drawer", "blotter"],
     "The answer must be used to absorb extra ink. Blotters are designed to absorb liquids. Therefore, the answer is blotter (e)."),
    ("What home entertainment equipment requires cable?",
     ["radio shack", "substation", "television", "cabinet", "desk"],
     "The answer must require cable. Cable is used to provide satellite channels to televisions. Therefore, the answer is television (c)."),
    ("If a person wanted to become a good writer what should they do in college?",
     ["word sentence", "own animal", "read the newspaper", "catch cold", "study literature"],
     "To become a good writer, one should engage in activities that improve writing skills and expose them to various writing styles. Studying literature in college provides exposure to diverse writing styles, techniques, and literary works. It also often involves writing essays and analyzing texts, which are crucial for developing writing skills. Therefore, the answer is study literature (e)."),
    ("What happens when I make a pig of myself eating breakfast?",
     ["full stomach", "gain energy", "dieting", "have energy", "feeling satisfied"],
     "The phrase 'make a pig of oneself' is an idiom meaning to eat excessively or greedily. When someone eats too much, especially during breakfast, the most immediate physical consequence is a feeling of fullness or having a full stomach. This is because the stomach has been filled to capacity or beyond. Therefore, the answer is full stomach (a)."),
    ("The king needed to feel safe, where did he go?",
     ["castle", "throne room", "deck of cards", "fort", "court"],
     "Historically, kings lived in castles for several reasons, including safety. Castles were designed as fortified residences, with thick walls, towers, and often surrounded by moats, providing maximum security against potential threats. The castle was not just a home but also a stronghold. Therefore, the answer is castle (a).")
]

star = ImprovedSTaR(few_shot_examples)

# Extended dataset with more diverse questions
dataset = [
    ("Where do you put your grapes just before checking out?",
     ["mouth", "grocery cart", "super market", "fruit basket", "fruit market"],
     "grocery cart"),
    ("Google Maps and other highway and street GPS services have replaced what?",
     ["united states", "mexico", "countryside", "atlas", "oceans"],
     "atlas"),
    ("What does hearing someone play violin beautifully make you?",
     ["buy earplugs", "inspiring", "guitar", "make music", "like music"],
     "inspiring"),
    ("What might someone get from learning about science?",
     ["headache", "see things differently", "increased knowledge", "accidents", "appreciation of nature"],
     "increased knowledge"),
    ("If ships are in a storm and the sun comes out, what has happened?",
     ["sinks", "cross river", "collide", "weather clears", "carry people"],
     "weather clears"),
    ("What emotion does committing murder induce?",
     ["fear", "go to jail", "problems", "guilt", "dead"],
     "guilt"),
    ("What can planning vacation lead to between a couple when they disagree?",
     ["relaxation", "enjoying", "arguments", "going abroad", "spending money"],
     "arguments")
]

# Run STaR algorithm
for iteration in range(5):  # 5 iterations of STaR
    print(f"Iteration {iteration + 1}")
    correct_count = 0
    for question, answer_choices, correct_answer in dataset:
        # Generate rationale and answer using knowledge from previous iterations
        rationale, predicted_answer, confidence = star.generate_rationale(question, answer_choices)
        print(f"Question: {question}")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Rationale: {rationale[:200]}...")  # Print first 200 characters of rationale
        
        if predicted_answer.lower() == correct_answer.lower():
            correct_count += 1
            # Add correct answers to the dataset for future fine-tuning
            star.dataset.append((question, str(answer_choices), f"{rationale} Therefore, the answer is {predicted_answer}."))
        else:
            # For incorrect answers, generate a rationalization to improve future performance
            rationalized_rationale = star.rationalize(question, answer_choices, correct_answer)
            star.dataset.append((question, str(answer_choices), f"{rationalized_rationale} Therefore, the answer is {correct_answer}."))
        print()
    
    # Simulate fine-tuning by updating few-shot examples
    star.finetune()
    accuracy = correct_count / len(dataset)
    print(f"Accuracy: {accuracy:.2f}")
    print("=" * 50)
    print()


# Test on new questions
new_questions = [
    ("What is the most likely outcome of regular exercise?",
     ["weight gain", "improved fitness", "loss of appetite", "increased stress", "decreased energy"],
     "improved fitness"),
    ("In a democratic society, what is the primary way citizens influence government policies?",
     ["writing letters", "protesting", "voting", "social media posts", "ignoring politics"],
     "voting"),
    ("What natural phenomenon is most closely associated with the water cycle?",
     ["earthquakes", "volcanic eruptions", "rain", "solar eclipses", "aurora borealis"],
     "rain")
]

for question, answer_choices, correct_answer in new_questions:
    # Generate rationale and answer for new questions using accumulated knowledge
    rationale, answer, confidence = star.generate_rationale(question, answer_choices)
    print(f"New Question: {question}")
    print(f"Predicted Answer: {answer}")
    print(f"Correct Answer: {correct_answer}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Rationale: {rationale}")
    print("=" * 50)
    print()