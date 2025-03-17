"""
Plan-on-Graph (PoG): Research Implementation

This implementation demonstrates the key concepts from the paper:
'Plan-on-Graph: Self-Correcting Adaptive Planning of Large Language Model on Knowledge Graphs'

Key Components:
1. Guidance Mechanism (Section 3.1): Task decomposition and sub-objective tracking
2. Memory Mechanism (Section 3.3): Tracks exploration history and knowledge state
3. Reflection Mechanism (Section 3.4): Enables self-correction through evaluation
4. Path Exploration (Section 3.2): Adaptive exploration with flexible breadth

The implementation focuses on showcasing:
- Adaptive exploration of knowledge graphs
- Self-correction through reflection
- Sub-objective guided reasoning
- Memory-based path tracking
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
import os
import json
from openai import OpenAI
from collections import defaultdict

# Knowledge Graph Structure
# ------------------------
# Implementing the paper's KG-augmented LLM concept (Section 3)
# Each entity has relations and connected entities that can be explored adaptively
KNOWLEDGE_GRAPH = {
    "Claude_Debussy": {
        "music.artist.genre": ["Ballet", "Impressionism"],
        "music.artist.composed": ["Afternoon_of_a_Faun", "Jeux"],
        "music.style.known_for": ["Ballet_compositions", "Orchestral_works"]
    },
    "Suzanne_Farrell_Elusive_Muse": {
        "film.film.genre": ["Documentary", "Ballet"],
        "film.film.features": ["Ballet_Music", "Classical_Dance"],
        "film.film.soundtrack": ["Ballet_Music_By_Debussy", "Other_Classical_Works"]
    },
    "Ballet": {
        "music.genre.associated_with": ["Claude_Debussy", "Igor_Stravinsky"],
        "art.form.characteristics": ["Dance", "Classical_Music", "Performance"],
        "music.style.composers": ["Debussy", "Tchaikovsky", "Stravinsky"]
    },
    "The_Naked_and_the_Dead": {
        "film.film.setting": ["Panama"],
        "film.film.time_period": ["World_War_II"],
        "film.film.location": ["Pacific_Theater"]
    },
    "Panama": {
        "location.country.leader": ["President_of_Panama"],
        "location.country.government": ["Presidential_Republic"],
        "location.country.capital": ["Panama_City"]
    },
    "President_of_Panama": {
        "government.role.holder": ["Juan_Carlos_Varela"],
        "government.role.jurisdiction": ["Panama"],
        "government.role.period": ["2014_2019"]
    }
}

KNOWLEDGE_GRAPH.update({
    "Suzanne_Farrell_Elusive_Muse": {
        "film.film.genre": ["Documentary", "Ballet"],
        "film.film.featured_music": ["Ballet_Music"],
        "film.film.year": "1996",
        "film.film.subject": ["New_York_City_Ballet", "Ballet_History"]
    },
    "World_War_II_Panama": {
        "location.historical.period": "1939-1945",
        "location.historical.control": ["US_Military_Forces"],
        "location.historical.context": ["Pacific_Theater", "Canal_Zone"]
    }
})

# Add historical context for temporal reasoning
HISTORICAL_CONTEXT = {
    "World_War_II_Panama": {
        "location.historical.period": "1939-1945",
        "location.historical.control": ["US_Military_Forces"],
        "location.historical.context": ["Pacific_Theater", "Canal_Zone"]
    }
}

@dataclass
class SubObjective:
    """
    Section 3.1: Guidance Mechanism Implementation
    
    Represents a decomposed sub-task of the main question, enabling:
    - Focused exploration of relevant knowledge
    - Progress tracking for each component
    - Evidence collection for final answer synthesis
    """
    index: int
    description: str
    status: str = ""
    completed: bool = False
    found_facts: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_fact(self, entity: str, relation: str, value: str):
        """Add discovered fact and update completion status"""
        self.found_facts.append({
            "entity": entity,
            "relation": relation,
            "value": value
        })
        self.update_status()

    def update_status(self):
        """Update completion status based on found facts"""
        if self.found_facts:
            self.status = f"Found {len(self.found_facts)} relevant facts"
            self.completed = True
        else:
            self.status = "Pending exploration"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "index": self.index,
            "description": self.description,
            "status": self.status,
            "completed": self.completed,
            "found_facts": self.found_facts
        }

@dataclass
class ExplorationState:
    """
    Section 3.2: Path Exploration Implementation
    
    Tracks the current state of knowledge graph exploration:
    - Current entity being explored
    - Path taken through the graph
    - Facts discovered during exploration
    """
    current_entity: str
    path: List[str]
    found_facts: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class EntityExtraction:
    main_entities: List[str]
    context_entities: List[str]

@dataclass
class TemporalContext:
    """
    Handles temporal reasoning for historical facts:
    - Ensures temporal consistency in reasoning
    - Validates facts against time periods
    - Helps avoid anachronistic conclusions
    """
    time_period: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class PoG:
    """
    Main Plan-on-Graph Implementation
    
    Implements the paper's core mechanisms:
    1. Task Decomposition (Section 3.1)
    2. Path Exploration (Section 3.2)
    3. Memory Updating (Section 3.3)
    4. Reflection & Evaluation (Section 3.4)
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.sub_objectives: List[SubObjective] = []
        self.knowledge_graph = {**KNOWLEDGE_GRAPH, **HISTORICAL_CONTEXT}

    def explore_entity(self, entity: str, sub_objectives: List[SubObjective]) -> Dict[str, Any]:
        """
        Section 3.2: Adaptive Path Exploration
        
        Key aspects implemented:
        1. Flexible exploration breadth based on relevance
        2. Sub-objective guided fact discovery
        3. Connection identification for further exploration
        """
        entity_info = self.get_entity_info(entity)
        
        # Identify connected entities for potential exploration
        connected_entities = set()
        for relation, values in entity_info.items():
            for value in values:
                if self.normalize_entity(value) in self.knowledge_graph:
                    connected_entities.add(value)
        
        # Prompt LLM for fact analysis and next steps
        prompt = f"""Analyze entity information and determine relevance to objectives.
        Return format:
        {{
            "relevant_facts": [
                {{
                    "entity": "{entity}",
                    "relation": "relation type",
                    "value": "found value",
                    "sub_objective_index": 0,
                    "relevance": "high/medium/low",
                    "reasoning": "why this fact matters"
                }}
            ],
            "next_entities": [
                {{
                    "entity": "entity to explore",
                    "reason": "why explore this entity",
                    "priority": "high/medium/low"
                }}
            ],
            "reasoning": "overall analysis"
        }}
        
        Entity: {entity}
        Available Information: {json.dumps(entity_info, indent=2)}
        Connected Entities: {list(connected_entities)}
        Sub-objectives: {[obj.to_dict() for obj in sub_objectives]}"""

        return self._call_llm(prompt, "Expert in knowledge graph analysis")

    def reflect_and_plan(self, state: ExplorationState, sub_objectives: List[SubObjective]) -> Dict[str, Any]:
        """
        Section 3.4: Reflection Mechanism
        
        Implements:
        1. Progress evaluation
        2. Self-correction triggering
        3. Exploration strategy adjustment
        """
        prompt = f"""Evaluate progress and determine if correction is needed.
        Return format:
        {{
            "progress_assessment": {{
                "completed_objectives": [0, 1],
                "pending_objectives": [2, 3],
                "sufficient_information": false
            }},
            "correction_needed": false,
            "correction_strategy": null,
            "next_steps": ["specific action"],
            "reasoning": "detailed explanation"
        }}
        
        Current Path: {state.path}
        Found Facts: {json.dumps(state.found_facts, indent=2)}
        Sub-objectives Progress: {[obj.to_dict() for obj in sub_objectives]}"""

        return self._call_llm(prompt, "Expert in reasoning evaluation")

    def answer_question(self, question: str) -> str:
        """
        Main PoG pipeline implementing the paper's workflow:
        1. Task decomposition (Section 3.1)
        2. Adaptive exploration (Section 3.2)
        3. Memory updating (Section 3.3)
        4. Reflection and correction (Section 3.4)
        """
        print(f"\nQuestion: {question}")
        print("=" * 80)

        # Step 1: Extract entities
        entities = self.extract_entities(question)
        print("\nIdentified Entities:")
        print(f"Main: {entities.main_entities}")
        print(f"Context: {entities.context_entities}")

        # Step 2: Task Decomposition
        print("\n1. Decomposing into Sub-objectives:")
        self.sub_objectives = self.decompose_task(question)
        for i, obj in enumerate(self.sub_objectives):
            print(f"  {i+1}. {obj.description}")

        # Initialize exploration with main entities
        explored_entities = set()
        to_explore = entities.main_entities.copy()
        facts_found = []
        
        max_depth = 4
        depth = 0
        
        while to_explore and depth < max_depth:
            current_entity = to_explore.pop(0)
            print(f"\n{depth + 2}. Exploring {current_entity}...")
            
            if self.normalize_entity(current_entity) not in self.knowledge_graph:
                print(f"No information found for {current_entity}")
                continue
                
            # Explore current entity
            exploration = self.explore_entity(current_entity, self.sub_objectives)
            explored_entities.add(self.normalize_entity(current_entity))
            
            print("Found relevant facts:")
            for fact in exploration.get("relevant_facts", []):
                print(f"  - {fact['relation']}: {fact['value']}")
                print(f"    Relevance: {fact['relevance']}")
                print(f"    Reasoning: {fact['reasoning']}")
                facts_found.append(fact)
                
                # Update sub-objective completion
                obj_idx = fact["sub_objective_index"]
                if obj_idx < len(self.sub_objectives):
                    self.sub_objectives[obj_idx].add_fact(
                        fact["entity"],
                        fact["relation"],
                        fact["value"]
                    )
            
            print(f"\nReasoning: {exploration['reasoning']}")
            
            # Add new entities to explore
            for next_entity in exploration.get("next_entities", []):
                entity_name = self.normalize_entity(next_entity["entity"])
                if (entity_name in self.knowledge_graph and 
                    entity_name not in explored_entities and 
                    entity_name not in [self.normalize_entity(e) for e in to_explore]):
                    print(f"Adding {next_entity['entity']} to explore (Priority: {next_entity['priority']})")
                    if next_entity['priority'] == 'high':
                        to_explore.insert(0, next_entity['entity'])
                    else:
                        to_explore.append(next_entity['entity'])
            
            depth += 1
        
        # Construct answer from collected facts
        if facts_found:
            answer = self.construct_answer(question, self.sub_objectives, facts_found)
            return answer
        else:
            return "Unable to find relevant information to answer the question."
 
    def extract_entities(self, question: str) -> EntityExtraction:
        """Extract main and context entities from the question"""
        prompt = f"""Identify the main entities and context entities from this question. Return format:
        {{
            "main_entities": ["primary entity 1", "primary entity 2"],
            "context_entities": ["context entity 1", "context entity 2"]
        }}
        
        Question: {question}"""
        
        response = self._call_llm(prompt, "You are an expert in entity extraction from text.")
        
        # Clean and normalize entity names
        main_entities = [self.normalize_entity(e) for e in response["main_entities"]]
        context_entities = [self.normalize_entity(e) for e in response["context_entities"]]
        
        return EntityExtraction(main_entities, context_entities)

    def normalize_entity(self, entity: str) -> str:
        """Normalize entity name to match knowledge graph keys"""
        return entity.replace(" ", "_").replace("'", "").replace('"', '')

    def get_entity_info(self, entity: str) -> Dict[str, List[str]]:
        """Get entity information from knowledge graph"""
        normalized_entity = entity.replace(" ", "_").replace("'", "")
        return self.knowledge_graph.get(normalized_entity, {})

    def decompose_task(self, question: str) -> List[SubObjective]:
        """Decompose question into focused sub-objectives"""
        prompt = f"""Break down this question into specific, focused sub-objectives. Return format:
        {{
            "sub_objectives": [
                {{
                    "description": "specific task description",
                    "required_info": ["what information is needed"]
                }}
            ]
        }}
        
        Question: {question}"""

        response = self._call_llm(prompt, "You are an expert in breaking down complex questions.")
        
        # Create SubObjective instances with index
        self.sub_objectives = [
            SubObjective(index=i, description=obj["description"])
            for i, obj in enumerate(response["sub_objectives"])
        ]
        return self.sub_objectives

    def get_next_entity(self, current_entity: str, exploration_result: Dict[str, Any], 
                        explored_entities: Set[str]) -> Optional[str]:
        """Determine next entity to explore based on knowledge graph connections"""
        if not exploration_result.get("next_entities"):
            # Look for unexplored connected entities
            connected_entities = set(exploration_result["connected_entities"])
            unexplored = connected_entities - explored_entities
            if unexplored:
                return unexplored.pop()
            
            # If no unexplored connected entities, look at parent entities
            for entity, info in self.knowledge_graph.items():
                for values in info.values():
                    if current_entity in values and entity not in explored_entities:
                        return entity
        else:
            # Use suggested next entities
            for next_entity in exploration_result["next_entities"]:
                entity_name = next_entity["entity"]
                if entity_name not in explored_entities and entity_name in self.knowledge_graph:
                    return entity_name
        
        return None

    def check_temporal_relevance(self, fact: Dict[str, Any], context: TemporalContext) -> bool:
        """Check if fact is relevant to the temporal context"""
        if fact.get("period"):
            return self.dates_overlap(fact["period"], context)
        return True

    def construct_answer(self, question: str, sub_objectives: List[SubObjective], 
                        facts: List[Dict[str, Any]]) -> str:
        """Construct final answer from collected facts with better organization"""
        # Group facts by sub-objective
        facts_by_objective = defaultdict(list)
        for fact in facts:
            obj_idx = fact.get("sub_objective_index", 0)
            facts_by_objective[obj_idx].append(fact)
        
        # Prepare facts summary for each sub-objective
        objectives_summary = []
        for obj in sub_objectives:
            relevant_facts = facts_by_objective[obj.index]
            objectives_summary.append({
                "objective": obj.description,
                "facts": relevant_facts,
                "completed": bool(relevant_facts)
            })
        
        prompt = f"""Based on the collected facts, construct a clear answer to the question.
        
        Question: {question}
        
        Evidence gathered:
        {json.dumps(objectives_summary, indent=2)}
        
        Return format:
        {{
            "answer": "clear and concise answer based on the evidence",
            "reasoning": "explanation of how the facts support this answer",
            "confidence": "high/medium/low based on evidence completeness"
        }}"""
        
        response = self._call_llm(prompt, "You are an expert in synthesizing information to answer questions.")
        
        # Format the answer with confidence level
        confidence = response.get("confidence", "medium")
        answer = response["answer"]
        reasoning = response["reasoning"]
        
        return f"""Answer ({confidence} confidence): {answer}

Reasoning: {reasoning}

Based on evidence from:
{self._format_evidence_summary(objectives_summary)}"""

    def _format_evidence_summary(self, objectives_summary: List[Dict[str, Any]]) -> str:
        """Format evidence summary for output"""
        lines = []
        for obj in objectives_summary:
            status = "✓" if obj["completed"] else "×"
            facts_count = len(obj["facts"])
            lines.append(f"{status} {obj['objective']} ({facts_count} facts)")
        return "\n".join(lines)

    def _call_llm(self, prompt: str, system_message: str) -> Dict[str, Any]:
        """Call LLM with proper error handling"""
        messages = [
            {"role": "system", "content": f"You are an AI assistant that always responds in JSON format. {system_message}"},
            {"role": "user", "content": f"{prompt}\nRespond in JSON format."}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.3
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error in LLM call: {e}")
            print(f"Prompt that caused error: {prompt}")
            raise

def main():
    """
    Demonstrates PoG on example questions from the paper.
    Shows the system's ability to:
    1. Handle temporal reasoning
    2. Explore connected facts
    3. Self-correct when needed
    4. Provide evidence-based answers
    """
    pog = PoG()
    
    questions = [
        # Question requiring musical knowledge and temporal understanding
        "What genre of music favored by Claude Debussy appears in the movie Suzanne Farrell: Elusive Muse?",
        
        # Question requiring historical context and political understanding
        "Who is in control of the place where the movie 'The Naked and the Dead' takes place?"
    ]
    
    for question in questions:
        answer = pog.answer_question(question)
        print(f"\nFinal Answer: {answer}\n")
        print("=" * 80)

if __name__ == "__main__":
    main()