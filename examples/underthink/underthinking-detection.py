"""
Implementation of underthinking detection and TIP (Thought Switching Penalty) decoding
Based on 'On the Underthinking of o1-Like LLMs' (Wang et al., 2025)

Key components:
1. Thought detection - Identify thought transitions in model responses
2. Underthinking metric - Measure token efficiency in incorrect responses 
3. TIP decoding - Apply penalties to discourage premature thought switching
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import numpy as np
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class ThoughtAnalysis:
    """Analysis of a single thought within a solution"""
    thought_text: str
    is_correct: bool
    confidence_score: float
    token_count: int
    thought_index: int

@dataclass 
class SolutionAnalysis:
    """Analysis of a complete solution attempt"""
    problem_id: str
    is_correct: bool
    thoughts: List[ThoughtAnalysis]
    total_tokens: int
    underthinking_score: float
    first_correct_thought_index: Optional[int]

@dataclass
class EvaluationResult:
    """Structured evaluation of a thought"""
    explanation: str
    confidence_score: float
    can_lead_to_solution: bool

class MathProblemDifficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3
    VERY_HARD = 4
    EXTREME = 5

@dataclass
class MathProblem:
    """Math problem with metadata"""
    problem_id: str
    text: str
    solution: str
    difficulty: MathProblemDifficulty
    category: str
    expected_steps: int

class UnderthinkingDetector:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model = model_name
        self.thought_markers = ["alternatively", "another approach", "let's try", "instead"]
        
    def detect_thoughts(self, solution_text: str) -> List[str]:
        """Split solution into distinct thoughts based on transition markers"""
        prompt = {
            "role": "system",
            "content": "You are a precise thought boundary detector for mathematical reasoning."
        }
        
        user_prompt = {
            "role": "user", 
            "content": f"""
            Split this solution into separate thoughts. Output in JSON format:
            {{
                "thoughts": [
                    {{"text": "thought_text", "start_index": int}}
                ]
            }}

            Solution: {solution_text}
            """
        }

        response = client.chat.completions.create(
            model=self.model,
            messages=[prompt, user_prompt],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return [t["text"] for t in result["thoughts"]]

    def evaluate_thought(self, problem: MathProblem, thought: str) -> EvaluationResult:
        """Evaluate if a thought could lead to correct solution"""
        prompt = {
            "role": "system",
            "content": "You are an expert evaluator of mathematical reasoning steps."
        }

        evaluation_prompt = {
            "role": "user",
            "content": f"""
            Problem: {problem.text}
            Expected Answer: {problem.solution}
            Solution Attempt: {thought}

            Please analyze if this solution attempt could lead to the correct answer.
            Output in JSON format:
            {{
                "explanation": "detailed analysis",
                "confidence_score": float between 0-2,
                "can_lead_to_solution": boolean
            }}
            """
        }

        response = client.chat.completions.create(
            model=self.model,
            messages=[prompt, evaluation_prompt],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return EvaluationResult(**result)

    def calculate_underthinking_score(self, analysis: SolutionAnalysis) -> float:
        """Calculate underthinking score based on token efficiency"""
        if analysis.is_correct or not analysis.first_correct_thought_index:
            return 0.0
            
        tokens_to_first_correct = sum(
            t.token_count for t in analysis.thoughts[:analysis.first_correct_thought_index + 1]
        )
        return 1 - (tokens_to_first_correct / analysis.total_tokens)

    def analyze_solution(self, problem: MathProblem, solution: str, is_correct: bool) -> SolutionAnalysis:
        """Analyze a solution attempt for underthinking"""
        thoughts = self.detect_thoughts(solution)
        
        thought_analyses = []
        first_correct_idx = None
        total_tokens = len(solution.split())
        
        for idx, thought in enumerate(thoughts):
            evaluation = self.evaluate_thought(problem, thought)
            
            analysis = ThoughtAnalysis(
                thought_text=thought,
                is_correct=evaluation.can_lead_to_solution,
                confidence_score=evaluation.confidence_score,
                token_count=len(thought.split()),
                thought_index=idx
            )
            
            if evaluation.can_lead_to_solution and first_correct_idx is None:
                first_correct_idx = idx
                
            thought_analyses.append(analysis)

        solution_analysis = SolutionAnalysis(
            problem_id=problem.problem_id,
            is_correct=is_correct,
            thoughts=thought_analyses,
            total_tokens=total_tokens,
            underthinking_score=0.0,
            first_correct_thought_index=first_correct_idx
        )
        
        solution_analysis.underthinking_score = self.calculate_underthinking_score(solution_analysis)
        return solution_analysis

class TIPDecoder:
    """Implements Thought Switching Penalty decoding strategy"""
    def __init__(self, penalty_strength: float = 3.0, penalty_duration: int = 600):
        self.alpha = penalty_strength  # Penalty strength
        self.beta = penalty_duration   # Penalty duration in tokens
        
    def adjust_logits(self, logits: List[float], 
                     position: int, 
                     thought_start: int,
                     thought_switch_tokens: List[int]) -> List[float]:
        """Apply TIP penalty to logits"""
        if position < thought_start + self.beta:
            for token_id in thought_switch_tokens:
                logits[token_id] -= self.alpha
        return logits

def generate_example_problems() -> List[MathProblem]:
    """Generate example math problems of varying difficulty"""
    return [
        MathProblem(
            problem_id="AIME_2024_1",
            text="Let a, b, x, and y be real numbers with a>4 and b>1 such that x²/a² + y²/(a²-16) = (x-20)²/(b²-1) + (y-11)²/b² = 1. Find the least possible value of a+b.",
            solution="15",
            difficulty=MathProblemDifficulty.VERY_HARD,
            category="Geometry",
            expected_steps=5
        ),
        MathProblem(
            problem_id="MATH500_ALG_1", 
            text="If 2x + 3y = 12 and 4x - y = 8, find the value of x + y.",
            solution="4",
            difficulty=MathProblemDifficulty.EASY,
            category="Algebra",
            expected_steps=2
        ),
        # Add more examples...
    ]

def main():
    """Demonstrate underthinking detection and TIP decoding"""
    detector = UnderthinkingDetector()
    tip_decoder = TIPDecoder()
    
    # Example problems
    problems = generate_example_problems()
    
    # Example incorrect solution with underthinking
    example_solution = """
    Let me try solving this using algebra. I'll substitute x² and y² terms...
    Alternatively, maybe this is a geometric problem about ellipses...
    Alternatively, I could try optimization with calculus...
    Perhaps I should use numerical methods instead...
    Let me guess, the answer is 15.
    """
    
    # Analyze solution
    analysis = detector.analyze_solution(problems[0], example_solution, False)
    
    print(f"Underthinking Score: {analysis.underthinking_score:.2f}")
    print(f"Number of thoughts: {len(analysis.thoughts)}")
    print(f"First correct thought at index: {analysis.first_correct_thought_index}")
    
    for idx, thought in enumerate(analysis.thoughts):
        print(f"\nThought {idx + 1}:")
        print(f"Correct path: {thought.is_correct}")
        print(f"Confidence: {thought.confidence_score}")
        print(f"Token count: {thought.token_count}")

if __name__ == "__main__":
    main()
