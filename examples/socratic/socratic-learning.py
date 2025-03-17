"""
Implementation of Socratic Learning with Language Games based on paper:
'Boundless Socratic Learning with Language Games' by Tom Schaul

This implementation demonstrates:
1. Language game framework
2. Recursive self-improvement through language interactions
3. Feedback alignment and coverage mechanisms
4. Meta-game scheduling
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from openai import OpenAI

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class GameResult:
    """Represents the outcome of a language game interaction"""
    score: float
    feedback: str
    learning_points: List[str]
    meta_improvements: List[str]
    knowledge_gained: Dict[str, Any]
    strategy_adaptations: List[str]

@dataclass
class GameConfig:
    """Configuration for a language game"""
    game_type: str
    difficulty: int
    rules: List[str]
    success_criteria: List[str]
    feedback_mechanism: str

@dataclass
class LanguageGameOutput:
    """Structured output for language game interactions"""
    game_id: str
    moves: List[str]
    reasoning: List[str]
    outcome: GameResult
    meta_learning: Dict[str, Any]

class GameType(Enum):
    DEBATE = "debate"
    PROOF_VERIFICATION = "proof_verification"
    CONCEPT_REFINEMENT = "concept_refinement"
    STRATEGY_FORMULATION = "strategy_formulation"
    META_LEARNING = "meta_learning"

@dataclass
class KnowledgeNode:
    """Represents a piece of knowledge and its evolution"""
    concept: str
    confidence: float
    sources: List[str]
    related_concepts: List[str]
    evolution_history: List[Dict[str, Any]]
    last_updated: datetime

@dataclass
class LearningStrategy:
    """Represents a learning strategy that can be modified"""
    name: str
    effectiveness: float
    application_count: int
    contexts: List[str]
    adaptations: List[Dict[str, Any]]

class SocraticAgent:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model = model_name
        self.game_history: List[LanguageGameOutput] = []
        self.meta_knowledge: Dict[str, Any] = {}
        self.knowledge_graph: Dict[str, KnowledgeNode] = {}
        self.learning_strategies: Dict[str, LearningStrategy] = {}
        self.performance_history: List[Dict[str, Any]] = []
        
    def generate_new_game(self) -> GameConfig:
        """
        Generate a new game based on past performance and knowledge state.
        Implements the game generation mechanism discussed in Section 6 of the paper.
        """
        # Create prompt for game generation
        knowledge_summary = self._summarize_knowledge_state()
        performance_patterns = self._analyze_performance_patterns()
        
        prompt = {
            "role": "user",
            "content": f"""
            Based on the following knowledge state and performance patterns, generate a new language game in JSON format:
            
            Knowledge State:
            {json.dumps(knowledge_summary, indent=2)}
            
            Performance Patterns:
            {json.dumps(performance_patterns, indent=2)}
            
            Generate a game that will:
            1. Address identified knowledge gaps
            2. Challenge current learning strategies
            3. Promote knowledge integration
            
            Response must follow this JSON structure:
            {{
                "game_type": "string",
                "difficulty": "integer",
                "rules": ["rule1", "rule2", ...],
                "success_criteria": ["criterion1", "criterion2", ...],
                "feedback_mechanism": "string"
            }}
            """
        }
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": "You are an expert in designing educational language games"}, prompt],
            response_format={"type": "json_object"}
        )
        
        game_spec = json.loads(response.choices[0].message.content)
        return GameConfig(**game_spec)

    def _summarize_knowledge_state(self) -> Dict[str, Any]:
        """Analyze current knowledge graph and identify gaps"""
        concepts = list(self.knowledge_graph.keys())
        avg_confidence = sum(node.confidence for node in self.knowledge_graph.values()) / len(self.knowledge_graph) if self.knowledge_graph else 0
        
        return {
            "total_concepts": len(concepts),
            "avg_confidence": avg_confidence,
            "weak_areas": [
                concept for concept, node in self.knowledge_graph.items()
                if node.confidence < 0.7
            ],
            "strong_areas": [
                concept for concept, node in self.knowledge_graph.items()
                if node.confidence > 0.9
            ]
        }

    def _analyze_performance_patterns(self) -> Dict[str, Any]:
        """Analyze performance history to identify patterns and areas for improvement"""
        if not self.performance_history:
            return {"status": "insufficient_data"}
            
        recent_games = self.performance_history[-10:]
        performance_by_type = {}
        
        for game in recent_games:
            game_type = game["game_type"]
            if game_type not in performance_by_type:
                performance_by_type[game_type] = []
            performance_by_type[game_type].append(game["score"])
        
        return {
            "performance_trends": {
                game_type: {
                    "avg_score": sum(scores) / len(scores),
                    "trend": "improving" if scores[-1] > scores[0] else "declining"
                }
                for game_type, scores in performance_by_type.items()
            },
            "recommended_focus": [
                game_type for game_type, scores in performance_by_type.items()
                if sum(scores) / len(scores) < 7.0
            ]
        }

    def adapt_learning_strategy(self, game_output: LanguageGameOutput):
        """
        Modify learning strategies based on game outcomes.
        Implements the strategy adaptation mechanism discussed in the paper.
        """
        # Analyze game performance
        strategy_prompt = {
            "role": "user",
            "content": f"""
            Based on this game outcome, suggest strategy adaptations in JSON format:
            
            Game Type: {game_output.game_id}
            Score: {game_output.outcome.score}
            Learning Points: {game_output.outcome.learning_points}
            
            Current Strategies:
            {json.dumps({name: asdict(strategy) for name, strategy in self.learning_strategies.items()}, indent=2)}
            
            Suggest strategy adaptations in this format:
            {{
                "new_strategies": [
                    {{"name": "strategy_name", "description": "strategy_description", "context": "context"}}
                ],
                "modifications": [
                    {{"strategy": "existing_strategy_name", "adaptation": "adaptation_description"}}
                ]
            }}
            """
        }
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": "You are an expert in learning strategy optimization"}, strategy_prompt],
            response_format={"type": "json_object"}
        )
        
        adaptations = json.loads(response.choices[0].message.content)
        
        # Implement new strategies
        for new_strategy in adaptations["new_strategies"]:
            self.learning_strategies[new_strategy["name"]] = LearningStrategy(
                name=new_strategy["name"],
                effectiveness=0.5,  # Initial effectiveness
                application_count=0,
                contexts=[new_strategy["context"]],
                adaptations=[{"description": new_strategy["description"], "timestamp": datetime.now().isoformat()}]
            )
        
        # Modify existing strategies
        for modification in adaptations["modifications"]:
            if modification["strategy"] in self.learning_strategies:
                strategy = self.learning_strategies[modification["strategy"]]
                strategy.adaptations.append({
                    "description": modification["adaptation"],
                    "timestamp": datetime.now().isoformat()
                })
                strategy.application_count += 1

    def play_language_game(self, game_config: GameConfig) -> LanguageGameOutput:
        """
        Play a language game according to specified configuration.
        Implements the core recursive self-improvement loop through language interaction.
        """
        # Construct system prompt based on game type and rules
        system_prompt = {
            "role": "system",
            "content": f"You are an expert in {game_config.game_type} with these rules: {', '.join(game_config.rules)}"
        }
        
        # Construct the game prompt with JSON format specification
        game_prompt = {
            "role": "user",
            "content": f"""
            Analyze and respond to this {game_config.game_type} scenario in JSON format:
            1. Consider the rules: {game_config.rules}
            2. Apply success criteria: {game_config.success_criteria}
            3. Provide structured reasoning and moves
            4. Include meta-learning insights
            
            Response must follow this JSON structure:
            {{
                "moves": ["move1", "move2", ...],
                "reasoning": ["reason1", "reason2", ...],
                "learning_points": ["point1", "point2", ...],
                "meta_improvements": ["improvement1", "improvement2", ...]
            }}
            """
        }

        # Execute the game interaction
        response = client.chat.completions.create(
            model=self.model,
            messages=[system_prompt, game_prompt],
            response_format={"type": "json_object"}
        )
        
        # Parse response and create structured output
        result = json.loads(response.choices[0].message.content)
        
        game_output = LanguageGameOutput(
            game_id=f"{game_config.game_type}_{datetime.now().isoformat()}",
            moves=result["moves"],
            reasoning=result["reasoning"],
            outcome=GameResult(
                score=self._calculate_score(result),
                feedback=self._generate_feedback(result),
                learning_points=result["learning_points"],
                meta_improvements=result["meta_improvements"],
                knowledge_gained=self._extract_knowledge(result),
                strategy_adaptations=self._extract_strategy_adaptations(result)
            ),
            meta_learning=self._extract_meta_learning(result)
        )
        
        # Update agent's meta-knowledge
        self._update_meta_knowledge(game_output)
        self.game_history.append(game_output)
        
        return game_output

    def _calculate_score(self, result: Dict[str, Any]) -> float:
        """Calculate game score based on moves and reasoning quality"""
        base_score = len(result["moves"]) + len(result["reasoning"])
        learning_bonus = len(result["learning_points"]) * 0.5
        meta_bonus = len(result["meta_improvements"]) * 0.3
        return min(10.0, base_score + learning_bonus + meta_bonus)

    def _generate_feedback(self, result: Dict[str, Any]) -> str:
        """Generate structured feedback for the game interaction"""
        return f"Completed {len(result['moves'])} moves with {len(result['learning_points'])} learning points"

    def _extract_meta_learning(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract meta-learning insights from game results"""
        return {
            "new_patterns": result.get("meta_improvements", []),
            "timestamp": datetime.now().isoformat(),
            "improvement_areas": [
                point for point in result.get("learning_points", [])
                if "improve" in point.lower() or "better" in point.lower()
            ]
        }

    def _extract_knowledge(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured knowledge from game results"""
        knowledge = {
            "concepts": [],
            "relations": [],
            "confidence_levels": {}
        }
        
        # Extract concepts from learning points
        for point in result["learning_points"]:
            # Add concepts mentioned in learning points
            words = point.lower().split()
            key_concepts = [w for w in words if len(w) > 5]  # Simple heuristic for important terms
            knowledge["concepts"].extend(key_concepts)
            
            # Add to knowledge graph with basic confidence
            for concept in key_concepts:
                if concept not in self.knowledge_graph:
                    self.knowledge_graph[concept] = KnowledgeNode(
                        concept=concept,
                        confidence=0.5,
                        sources=[result.get("game_type", "unknown")],
                        related_concepts=[],
                        evolution_history=[{
                            "timestamp": datetime.now().isoformat(),
                            "event": "concept_introduction",
                            "context": point
                        }],
                        last_updated=datetime.now()
                    )
                knowledge["confidence_levels"][concept] = 0.5

        return knowledge

    def _extract_strategy_adaptations(self, result: Dict[str, Any]) -> List[str]:
        """Extract strategy adaptations from game results"""
        adaptations = []
        
        # Extract strategy insights from meta improvements
        for improvement in result["meta_improvements"]:
            if any(keyword in improvement.lower() for keyword in ["strategy", "approach", "method", "technique"]):
                adaptations.append(improvement)

        return adaptations

    def _update_meta_knowledge(self, game_output: LanguageGameOutput):
        """Update agent's meta-knowledge based on game outcomes"""
        self.meta_knowledge.update({
            f"game_{game_output.game_id}": {
                "success_patterns": game_output.outcome.learning_points,
                "improvements": game_output.outcome.meta_improvements,
                "knowledge_gained": game_output.outcome.knowledge_gained,
                "strategy_adaptations": game_output.outcome.strategy_adaptations
            }
        })

def create_example_games() -> List[GameConfig]:
    """Create a diverse set of example language games"""
    return [
        # 1. Mathematical Proof Verification Game
        GameConfig(
            game_type=GameType.PROOF_VERIFICATION.value,
            difficulty=3,
            rules=[
                "Verify the logical consistency of the proof",
                "Identify any gaps in reasoning",
                "Suggest improvements for clarity"
            ],
            success_criteria=[
                "All logical steps are valid",
                "No missing intermediate steps",
                "Clear and precise language"
            ],
            feedback_mechanism="structured_validation"
        ),
        
        # 2. Concept Refinement Game
        GameConfig(
            game_type=GameType.CONCEPT_REFINEMENT.value,
            difficulty=2,
            rules=[
                "Define the initial concept clearly",
                "Identify potential ambiguities",
                "Propose refined definitions"
            ],
            success_criteria=[
                "Increased precision in definition",
                "Reduced ambiguity",
                "Practical applicability"
            ],
            feedback_mechanism="comparative_analysis"
        ),
        
        # 3. Strategic Debate Game
        GameConfig(
            game_type=GameType.DEBATE.value,
            difficulty=4,
            rules=[
                "Present clear arguments",
                "Address counterpoints",
                "Maintain logical consistency"
            ],
            success_criteria=[
                "Strong argument structure",
                "Effective counterargument handling",
                "Evidence-based reasoning"
            ],
            feedback_mechanism="peer_review"
        ),
        
        # 4. Meta-Learning Game
        GameConfig(
            game_type=GameType.META_LEARNING.value,
            difficulty=5,
            rules=[
                "Analyze previous game patterns",
                "Identify improvement opportunities",
                "Propose strategic adjustments"
            ],
            success_criteria=[
                "Pattern recognition",
                "Strategic insights",
                "Actionable improvements"
            ],
            feedback_mechanism="self_evaluation"
        )
    ]

def main():
    """
    Demonstrate the Socratic learning system with various language games,
    including dynamic game generation and strategy adaptation
    """
    # Initialize the agent
    agent = SocraticAgent()
    
    # Start with some initial games
    initial_games = create_example_games()
    
    print("Phase 1: Initial Learning")
    print("-----------------------")
    # Play initial games
    for game_config in initial_games:
        print(f"\nPlaying {game_config.game_type} game...")
        result = agent.play_language_game(game_config)
        agent.adapt_learning_strategy(result)
        
        print(f"Game completed with score: {result.outcome.score}")
        print(f"Learning points: {result.outcome.learning_points}")
        print(f"Meta-improvements: {result.outcome.meta_improvements}")
        
    print("\nPhase 2: Dynamic Game Generation")
    print("-------------------------------")
    # Generate and play new games based on performance
    for _ in range(2):  # Generate 2 new games
        new_game = agent.generate_new_game()
        print(f"\nGenerated new game of type: {new_game.game_type}")
        print(f"Rules: {new_game.rules}")
        print(f"Success criteria: {new_game.success_criteria}")
        
        result = agent.play_language_game(new_game)
        agent.adapt_learning_strategy(result)
        
        print(f"Game completed with score: {result.outcome.score}")
        print(f"Learning points: {result.outcome.learning_points}")
        print(f"Strategy adaptations: {result.outcome.strategy_adaptations}")
    
    print("\nPhase 3: Learning Strategy Evolution")
    print("----------------------------------")
    for strategy_name, strategy in agent.learning_strategies.items():
        print(f"\nStrategy: {strategy_name}")
        print(f"Effectiveness: {strategy.effectiveness}")
        print(f"Applications: {strategy.application_count}")
        print("Evolution:")
        for adaptation in strategy.adaptations:
            print(f"- {adaptation['description']} ({adaptation['timestamp']})")
    
    print("\nPhase 4: Knowledge State Analysis")
    print("--------------------------------")
    knowledge_summary = agent._summarize_knowledge_state()
    print("\nKnowledge Summary:")
    print(f"Total concepts: {knowledge_summary['total_concepts']}")
    print(f"Average confidence: {knowledge_summary['avg_confidence']:.2f}")
    print(f"Strong areas: {knowledge_summary['strong_areas']}")
    print(f"Areas needing improvement: {knowledge_summary['weak_areas']}")

if __name__ == "__main__":
    main()
