import os
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import json
import datetime
from enum import Enum
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

class CoALAPatternSolver:
    def __init__(self):
        self.memories = {
            MemoryType.WORKING: {
                "current_sequence": [],
                "analysis_state": {},
                "transformations": [],
                "pattern_hypotheses": []
            },
            MemoryType.SEMANTIC: {
                "pattern_examples": {
                    "arithmetic": [
                        {"sequence": [2, 4, 6, 8, 10], "formula": "a(n) = a(1) + (n-1)d"},
                        {"sequence": [1, 5, 9, 13, 17], "formula": "a(n) = a(1) + (n-1)d"}
                    ],
                    "geometric": [
                        {"sequence": [2, 4, 8, 16, 32], "formula": "a(n) = a(1) * r^(n-1)"},
                        {"sequence": [3, 9, 27, 81, 243], "formula": "a(n) = a(1) * r^(n-1)"}
                    ],
                    "fibonacci": [
                        {"sequence": [1, 1, 2, 3, 5, 8], "formula": "a(n) = a(n-1) + a(n-2)"},
                        {"sequence": [2, 3, 5, 8, 13, 21], "formula": "a(n) = a(n-1) + a(n-2)"}
                    ],
                    "interleaved": [
                        {"sequence": [2, 3, 4, 7, 8, 11], "formula": "even: a(n)=2n, odd: a(n)=2n+1"},
                        {"sequence": [2, 3, 8, 7, 32, 11], "formula": "even: 2^n, odd: arithmetic"}
                    ],
                    "position_based": [
                        {"sequence": [1, 4, 13, 40, 121], "formula": "a(n) = n * a(n-1)"},
                        {"sequence": [2, 6, 24, 120, 720], "formula": "a(n) = n * a(n-1)"}
                    ]
                }
            },
            MemoryType.EPISODIC: {
                "pattern_history": [],
                "error_cases": [],
                "successful_patterns": []
            }
        }

    def _detect_pattern_type(self, sequence: List[int], transformations: Dict) -> Dict:
        """Use LLM to detect pattern type with memory-enhanced context"""
        
        # Retrieve similar patterns from episodic memory
        similar_patterns = self._find_similar_patterns(sequence)
        
        # Get relevant examples from semantic memory
        semantic_examples = []
        for pattern_type, examples in self.memories[MemoryType.SEMANTIC]["pattern_examples"].items():
            if similar_patterns and any(p["pattern_type"].lower() == pattern_type.lower() 
                                    for p in similar_patterns):
                semantic_examples.extend(examples)
        
        # Format memory examples
        similar_patterns_json = json.dumps([{
            'sequence': p['sequence'],
            'pattern_type': p['pattern_type'],
            'formula': p['formula']
        } for p in similar_patterns], indent=2) if similar_patterns else "No similar patterns found"
        
        semantic_examples_json = json.dumps(semantic_examples, indent=2) if semantic_examples else "No directly relevant examples"
        
        successful_patterns_json = json.dumps([{
            'sequence': p['sequence'],
            'pattern_type': p['analysis']['pattern_type'],
            'formula': p['analysis']['formula']
        } for p in self.memories[MemoryType.EPISODIC]['successful_patterns'][-3:]], indent=2)
        
        # Create expected response format template
        response_format = {
            "primary_pattern": {
                "type": "pattern type name",
                "formula": "mathematical formula",
                "explanation": "detailed explanation",
                "similar_to": "reference to similar pattern if applicable"
            },
            "sub_patterns": [
                {
                    "type": "sub-pattern type",
                    "formula": "sub-pattern formula",
                    "applies_to": "which part of sequence"
                }
            ],
            "confidence": "float between 0 and 1",
            "verification_steps": [
                "step 1 to verify pattern",
                "step 2 to verify pattern"
            ]
        }
        
        # Create memory-enhanced prompt
        prompt = f"""Given a sequence and its transformations, identify the pattern type and formula.

    Sequence to analyze: {sequence}

    Transformations Analysis:
    1. First differences: {transformations['differences']}
    2. Second differences: {transformations['second_differences']}
    3. Position split:
    - Even positions: {transformations['position_split']['even']}
    - Odd positions: {transformations['position_split']['odd']}
    4. Position-based patterns:
    - Position correlations: {transformations['position_correlations']}
    - Position differences: {transformations['position_diffs']}
    5. Modulo patterns:
    - Mod 2: {transformations['modulo_patterns'][2]}
    - Mod 3: {transformations['modulo_patterns'][3]}
    - Mod 4: {transformations['modulo_patterns'][4]}

    Similar patterns found in memory:
    {similar_patterns_json}

    Relevant pattern examples:
    {semantic_examples_json}

    Previous successful patterns with high confidence:
    {successful_patterns_json}

    Return a detailed analysis in the following JSON format:
    {json.dumps(response_format, indent=2)}

    If similar patterns are found, use them to inform your analysis. Consider how past successful patterns might apply to the current sequence. Favor pattern types that have worked well in similar cases."""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in mathematical pattern analysis with memory of past successes."},
                    {"role": "user", "content": prompt}
                ],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "analyze_pattern_type",
                        "description": "Analyze and identify numerical sequence pattern type",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "primary_pattern": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string"},
                                        "formula": {"type": "string"},
                                        "explanation": {"type": "string"},
                                        "similar_to": {"type": "string"}
                                    },
                                    "required": ["type", "formula", "explanation"]
                                },
                                "sub_patterns": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "type": {"type": "string"},
                                            "formula": {"type": "string"},
                                            "applies_to": {"type": "string"}
                                        }
                                    }
                                },
                                "confidence": {"type": "number"},
                                "verification_steps": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["primary_pattern", "confidence", "verification_steps"]
                        }
                    }
                }],
                tool_choice={"type": "function", "function": {"name": "analyze_pattern_type"}}
            )

            analysis = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            
            # Adjust confidence based on memory matches
            if similar_patterns:
                max_similarity = max(p["similarity"] for p in similar_patterns)
                analysis["confidence"] = min(1.0, analysis["confidence"] + (max_similarity * 0.1))
            
            # Store in working memory
            self.memories[MemoryType.WORKING]["pattern_hypotheses"].append(analysis)
            
            return analysis
            
        except Exception as e:
            print(f"Error in pattern detection: {str(e)}")
            return {
                "primary_pattern": {
                    "type": "unknown",
                    "formula": "undefined",
                    "explanation": "Pattern detection failed"
                },
                "confidence": 0.0,
                "verification_steps": []
            }

    def analyze_pattern(self, sequence: List[int]) -> Dict:
        """Main pattern analysis method using CoALA framework"""
        
        # Update working memory
        self.memories[MemoryType.WORKING]["current_sequence"] = sequence
        
        # Apply transformations
        transformations = self._apply_transformations(sequence)
        self.memories[MemoryType.WORKING]["transformations"] = transformations
        
        # Get pattern type using LLM
        pattern_analysis = self._detect_pattern_type(sequence, transformations)
        
        # Find similar patterns in episodic memory
        similar_patterns = self._find_similar_patterns(sequence)
        
        # Prepare final analysis
        next_terms = self._predict_next_terms(sequence, pattern_analysis, transformations)
        
        result = {
            "pattern_type": pattern_analysis["primary_pattern"]["type"],
            "formula": pattern_analysis["primary_pattern"]["formula"],
            "component_patterns": [
                {
                    "type": p["type"],
                    "formula": p["formula"]
                }
                for p in pattern_analysis.get("sub_patterns", [])
            ],
            "next_terms": next_terms,
            "explanation": pattern_analysis["primary_pattern"]["explanation"],
            "confidence": pattern_analysis["confidence"]
        }
        
        # Store in episodic memory if confidence is high
        if result["confidence"] > 0.8:
            self._store_analysis(sequence, result)
        
        return result

    def _predict_next_terms(self, sequence: List[int], pattern_analysis: Dict, transformations: Dict) -> List[int]:
        """Use LLM to predict next terms based on pattern analysis"""
        
        prompt = f"""Given the sequence and its pattern analysis, predict the next 3 terms.

Sequence: {sequence}
Pattern Type: {pattern_analysis['primary_pattern']['type']}
Formula: {pattern_analysis['primary_pattern']['formula']}
Sub-patterns: {pattern_analysis.get('sub_patterns', [])}

Return exactly three numbers in JSON format:
{{"next_terms": [number, number, number]}}"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in mathematical pattern continuation."},
                    {"role": "user", "content": prompt}
                ],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "predict_next_terms",
                        "description": "Predict next three terms in sequence",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "next_terms": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "minItems": 3,
                                    "maxItems": 3
                                }
                            },
                            "required": ["next_terms"]
                        }
                    }
                }],
                tool_choice={"type": "function", "function": {"name": "predict_next_terms"}}
            )

            prediction = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
            return prediction["next_terms"]
            
        except Exception as e:
            print(f"Error in term prediction: {str(e)}")
            return sequence[-3:]  # Return last 3 terms as fallback

    def _apply_transformations(self, sequence: List[int]) -> Dict:
        """Calculate sequence transformations"""
        differences = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]
        second_differences = [differences[i+1] - differences[i] for i in range(len(differences)-1)]
        
        return {
            "differences": differences,
            "second_differences": second_differences,
            "position_split": {
                "even": sequence[::2],
                "odd": sequence[1::2]
            },
            "position_diffs": [
                sequence[i] - sequence[i-1] if i > 0 else 0
                for i in range(len(sequence))
            ],
            "position_correlations": {
                "correlation_with_index": np.corrcoef(
                    range(len(sequence)), 
                    sequence
                )[0,1] if len(sequence) > 1 else 0
            },
            "modulo_patterns": {
                mod: [x % mod for x in sequence]
                for mod in [2, 3, 4]
            }
        }

    def _find_similar_patterns(self, sequence: List[int]) -> List[Dict]:
        """Find similar patterns in episodic memory"""
        similar_patterns = []
        
        for memory in self.memories[MemoryType.EPISODIC]["pattern_history"]:
            if len(memory["sequence"]) >= 3:
                similarity = self._calculate_sequence_similarity(
                    sequence[:min(len(sequence), len(memory["sequence"]))],
                    memory["sequence"][:min(len(sequence), len(memory["sequence"]))]
                )
                if similarity > 0.7:
                    similar_patterns.append({
                        "sequence": memory["sequence"],
                        "pattern_type": memory["analysis"]["pattern_type"],
                        "formula": memory["analysis"]["formula"],
                        "similarity": similarity
                    })
        
        return similar_patterns

    def _calculate_sequence_similarity(self, seq1: List[int], seq2: List[int]) -> float:
        """Calculate similarity between two sequences"""
        if not seq1 or not seq2:
            return 0.0
            
        # Normalize sequences
        max_val = max(max(seq1), max(seq2))
        if max_val == 0:
            return 1.0 if seq1 == seq2 else 0.0
            
        norm_seq1 = [x/max_val for x in seq1]
        norm_seq2 = [x/max_val for x in seq2]
        
        # Calculate difference-based similarity
        differences = [abs(a - b) for a, b in zip(norm_seq1, norm_seq2)]
        return 1 - (sum(differences) / len(differences))

    def _store_analysis(self, sequence: List[int], analysis: Dict):
        """Store analysis in episodic memory"""
        memory_entry = {
            "sequence": sequence,
            "analysis": analysis,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        self.memories[MemoryType.EPISODIC]["pattern_history"].append(memory_entry)
        if analysis["confidence"] > 0.9:
            self.memories[MemoryType.EPISODIC]["successful_patterns"].append(memory_entry)

def test_coala_solver():
    """Test function demonstrating memory-enhanced pattern detection"""
    solver = CoALAPatternSolver()
    
    test_sequences = [
        # Simple patterns to bootstrap memory
        [2, 4, 6, 8, 10, 12],                    # Simple arithmetic
        [2, 4, 8, 16, 32, 64],                   # Simple geometric
        [1, 1, 2, 3, 5, 8, 13],                  # Simple Fibonacci
        
        # Complex patterns to test memory utilization
        [2, 3, 8, 7, 32, 11, 128, 15, 512, 19],  # Interleaved power of 2 and arithmetic
        [1, 1, 3, 5, 11, 19, 37, 61, 115, 187],  # Modified Fibonacci
        [2, 5, 11, 14, 20, 23, 29, 32, 38, 41],  # Alternating increments
        [1, 4, 13, 40, 121, 364, 1093],          # Position-based multiplication
        [4, 6, 13, 28, 61, 132, 283],            # Nested differences
        
        # Test sequences similar to previous ones
        [2, 3, 4, 7, 8, 11, 16, 15],             # Similar to first interleaved
        [2, 2, 6, 10, 22, 38, 74],               # Similar to modified Fibonacci
        [3, 7, 14, 18, 25, 29, 36]               # Similar to alternating increments
    ]

    ground_truth = [
        # Sequence 1: Simple arithmetic (+2 each time)
        [2, 4, 6, 8, 10, 12, 14, 16, 18],
        
        # Sequence 2: Simple geometric (*2 each time)
        [2, 4, 8, 16, 32, 64, 128, 256, 512],
        
        # Sequence 3: Simple Fibonacci (sum of the previous two terms)
        [1, 1, 2, 3, 5, 8, 13, 21, 34, 55],
        
        # Sequence 4: Interleaved power of 2 and arithmetic
        [2, 3, 8, 7, 32, 11, 128, 15, 512, 19, 2048, 23, 8192],
        
        # Sequence 5: Modified Fibonacci (approximate growth pattern)
        [1, 1, 3, 5, 11, 19, 37, 61, 115, 187, 283, 499, 787],
        
        # Sequence 6: Alternating increments (+3 and +6 alternately)
        [2, 5, 11, 14, 20, 23, 29, 32, 38, 41, 47, 50, 56],
        
        # Sequence 7: Position-based multiplication (prev * 3 + 1)
        [1, 4, 13, 40, 121, 364, 1093, 3280, 9841, 29524],
        
        # Sequence 8: Nested differences (pattern in differences)
        [4, 6, 13, 28, 61, 132, 283, 556, 1031, 1830],
        
        # Sequence 9: Similar to first interleaved sequence
        [2, 3, 4, 7, 8, 11, 16, 15, 32, 19, 64],
        
        # Sequence 10: Similar to modified Fibonacci (complex pattern)
        [2, 2, 6, 10, 22, 38, 74, 166, 298, 478],
        
        # Sequence 11: Similar to alternating increments (+4 and +7 alternately)
        [3, 7, 14, 18, 25, 29, 36, 40, 47, 51],
    ]
    print("Testing CoALA Pattern Solver with Memory Enhancement\n")
    
    for i, sequence in enumerate(test_sequences):
        print(f"\n{'='*80}")
        print(f"Test Case {i+1}: {sequence}")
        print(f"{'='*80}")
        
        # Get similar patterns before analysis
        similar_patterns = solver._find_similar_patterns(sequence)
        if similar_patterns:
            print("\nFound similar patterns in memory:")
            for pattern in similar_patterns:
                print(f"- Sequence: {pattern['sequence']}")
                print(f"  Pattern type: {pattern['pattern_type']}")
                print(f"  Formula: {pattern['formula']}")
                print(f"  Similarity score: {pattern['similarity']:.3f}")
        
        # Perform analysis
        result = solver.analyze_pattern(sequence)
        
        print("\nAnalysis Results:")
        print(f"Pattern type: {result['pattern_type']}")
        print(f"Formula: {result['formula']}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        if result['component_patterns']:
            print("\nComponent Patterns:")
            for comp in result['component_patterns']:
                print(f"- Type: {comp['type']}")
                print(f"  Formula: {comp['formula']}")
        
        print("\nPredicted next terms:", result['next_terms'])
        print("\nExplanation:", result['explanation'])
        
        # Show memory utilization
        memory_stats = {
            "episodic": len(solver.memories[MemoryType.EPISODIC]["pattern_history"]),
            "successful": len(solver.memories[MemoryType.EPISODIC]["successful_patterns"])
        }
        print(f"\nMemory Statistics:")
        print(f"- Patterns in episodic memory: {memory_stats['episodic']}")
        print(f"- Successful patterns stored: {memory_stats['successful']}")
        
        # Verify predictions if possible
        actual_next = ground_truth[i][-3:]
        predicted_next = result['next_terms']
        print("\nPrediction Analysis:")
        print(f"Predicted next 3 terms: {predicted_next}")
        print(f"Actual next 3 terms: {actual_next}")
        if predicted_next == actual_next:
            print("✓ Prediction matches exactly!")
        else:
            print("× Prediction differs from actual")


if __name__ == "__main__":
    test_coala_solver()