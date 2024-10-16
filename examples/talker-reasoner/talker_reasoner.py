import os
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class BeliefState(BaseModel):
    coaching_phase: str
    primary_sleep_concern: str
    goals: List[str] = Field(default_factory=list)
    habits: List[str] = Field(default_factory=list)
    barriers: List[str] = Field(default_factory=list)
    sleep_environment: Dict[str, Any] = Field(default_factory=dict)
    understanding_progress: float = 0.0
    goal_setting_progress: float = 0.0
    planning_progress: float = 0.0

class TalkerReasoner:
    def __init__(self):
        self.memory = {"belief_state": BeliefState(
            coaching_phase="UNDERSTANDING",
            primary_sleep_concern="",
            goals=[],
            habits=[],
            barriers=[],
            sleep_environment={},
            understanding_progress=0.0,
            goal_setting_progress=0.0,
            planning_progress=0.0
        )}
        self.interaction_history = []

    def talker(self, user_input: str) -> Dict[str, Any]:
        """Implement the Talker agent using GPT-4o-mini"""
        prompt = f"""You are an AI sleep coach. Your task is to interact with the user in a conversational manner.
        Current belief state: {json.dumps(self.memory['belief_state'].model_dump())}
        Interaction history: {json.dumps(self.interaction_history)}
        User input: {user_input}
        
        Respond to the user in a helpful and empathetic way, based on the current belief state and interaction history.
        If you believe the current phase is complete and we should move to the next phase, set 'wait_for_reasoner' to true.
        
        Provide your response in JSON format with 'response' and 'wait_for_reasoner' keys."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert sleep coach AI assistant. Provide your responses in JSON format."},
                {"role": "user", "content": prompt}
            ],
            tool_choice={"type": "function", "function": {"name": "talker_response"}},
            tools=[{
                "type": "function",
                "function": {
                    "name": "talker_response",
                    "description": "Provide the Talker's response",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "response": {"type": "string"},
                            "wait_for_reasoner": {"type": "boolean"}
                        },
                        "required": ["response", "wait_for_reasoner"]
                    }
                }
            }]
        )
        
        return json.loads(response.choices[0].message.tool_calls[0].function.arguments)

    def mini_reasoner_understanding(self, user_input: str, talker_response: str) -> Dict[str, Any]:
        """Implement the Understanding phase Reasoner"""
        print("\n[DEBUG] Using mini-reasoner: UNDERSTANDING")
        prompt = f"""You are an AI sleep coach reasoner in the UNDERSTANDING phase. Update the belief state based on the user input and talker response.
        Current belief state: {json.dumps(self.memory['belief_state'].model_dump())}
        User input: {user_input}
        Talker response: {talker_response}
        
        Focus on understanding the user's primary sleep concerns and current habits. Update the understanding_progress (0.0 to 1.0) based on how complete our understanding is.
        If understanding_progress reaches 1.0, change the coaching_phase to GOAL_SETTING.
        Provide your response in JSON format, including all fields of the belief state."""

        return self._call_reasoner(prompt)

    def mini_reasoner_goal_setting(self, user_input: str, talker_response: str) -> Dict[str, Any]:
        """Implement the Goal Setting phase Reasoner"""
        print("\n[DEBUG] Using mini-reasoner: GOAL_SETTING")
        prompt = f"""You are an AI sleep coach reasoner in the GOAL_SETTING phase. Update the belief state based on the user input and talker response.
        Current belief state: {json.dumps(self.memory['belief_state'].model_dump())}
        User input: {user_input}
        Talker response: {talker_response}
        
        Focus on setting specific, measurable, achievable, relevant, and time-bound (SMART) sleep goals. Update the goal_setting_progress (0.0 to 1.0) based on how complete our goal setting is.
        If goal_setting_progress reaches 1.0, change the coaching_phase to PLANNING.
        Provide your response in JSON format, including all fields of the belief state."""

        return self._call_reasoner(prompt)

    def mini_reasoner_planning(self, user_input: str, talker_response: str) -> Dict[str, Any]:
        """Implement the Planning phase Reasoner"""
        print("\n[DEBUG] Using mini-reasoner: PLANNING")
        prompt = f"""You are an AI sleep coach reasoner in the PLANNING phase. Update the belief state based on the user input and talker response.
        Current belief state: {json.dumps(self.memory['belief_state'].model_dump())}
        User input: {user_input}
        Talker response: {talker_response}
        
        Develop a detailed sleep improvement plan based on the user's goals and barriers. Update the planning_progress (0.0 to 1.0) based on how complete our planning is.
        Provide your response in JSON format, including all fields of the belief state."""

        return self._call_reasoner(prompt)

    def _call_reasoner(self, prompt: str) -> Dict[str, Any]:
        """Helper method to call the Reasoner"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert sleep coach AI reasoner. Provide your responses in JSON format."},
                {"role": "user", "content": prompt}
            ],
            tool_choice={"type": "function", "function": {"name": "update_belief_state"}},
            tools=[{
                "type": "function",
                "function": {
                    "name": "update_belief_state",
                    "description": "Update the belief state",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "coaching_phase": {"type": "string"},
                            "primary_sleep_concern": {"type": "string"},
                            "goals": {"type": "array", "items": {"type": "string"}},
                            "habits": {"type": "array", "items": {"type": "string"}},
                            "barriers": {"type": "array", "items": {"type": "string"}},
                            "sleep_environment": {"type": "object"},
                            "understanding_progress": {"type": "number"},
                            "goal_setting_progress": {"type": "number"},
                            "planning_progress": {"type": "number"}
                        },
                        "required": ["coaching_phase", "primary_sleep_concern", "goals", "habits", "barriers", "sleep_environment", "understanding_progress", "goal_setting_progress", "planning_progress"]
                    }
                }
            }]
        )

        return json.loads(response.choices[0].message.tool_calls[0].function.arguments)

    def reasoner(self, user_input: str, talker_response: str) -> None:
        """Implement the hierarchical Reasoner agent"""
        current_phase = self.memory["belief_state"].coaching_phase
        
        if current_phase == "UNDERSTANDING":
            updated_belief = self.mini_reasoner_understanding(user_input, talker_response)
        elif current_phase == "GOAL_SETTING":
            updated_belief = self.mini_reasoner_goal_setting(user_input, talker_response)
        elif current_phase == "PLANNING":
            updated_belief = self.mini_reasoner_planning(user_input, talker_response)
        else:
            raise ValueError(f"Invalid coaching phase: {current_phase}")

        self.memory["belief_state"] = BeliefState(**updated_belief)

    def interact(self, user_input: str) -> str:
        talker_output = self.talker(user_input)
        
        if talker_output["wait_for_reasoner"]:
            print("\n[DEBUG] Talker is waiting for Reasoner")
            self.reasoner(user_input, talker_output["response"])
            print("\n[DEBUG] Reasoner finished, Talker generating new response")
            talker_output = self.talker(user_input)

        self.reasoner(user_input, talker_output["response"])
        self.interaction_history.append({"user": user_input, "ai": talker_output["response"]})
        return talker_output["response"]

# Example usage
agent = TalkerReasoner()

# List of example interactions to showcase the workflow
examples = [
    "Hi, I'm having trouble sleeping. Can you help?",
    "I think noises and light are affecting my sleep.",
    "Can you suggest some relaxation techniques?",
    "I've tried deep breathing, but it doesn't seem to help.",
    "What about my diet? Could that be affecting my sleep?",
    "I usually have coffee in the evening. Is that bad?",
    "How can I create a better sleep environment?",
    "What's a good bedtime routine?",
    "I often wake up in the middle of the night. Any advice?",
    "How much sleep should I be getting?",
    "Can you help me set some sleep goals?",
    "What's the ideal room temperature for sleeping?",
    "Are naps good or bad for nighttime sleep?",
    "I use my phone before bed. Is that okay?",
    "How can I stop worrying about work when trying to sleep?",
    "What's your opinion on sleep tracking apps?",
    "Can exercise help with sleep? When's the best time to exercise?",
    "I sleep better on weekends. How can I improve weekday sleep?",
    "Are there any foods that can help with sleep?",
    "Can you summarize the main points we've discussed about improving my sleep?"
]

# Run the examples
for i, example in enumerate(examples, 1):
    print(f"\nInteraction {i}:")
    print(f"User: {example}")
    response = agent.interact(example)
    print(f"AI: {response}")
    print(f"Updated Belief State: {json.dumps(agent.memory['belief_state'].model_dump(), indent=2)}")

# Reflection on the implementation
reflection = """
This improved implementation enhances the Talker-Reasoner architecture with the following key features:

1. Dynamic Phase Transitions: The system now uses progress tracking (understanding_progress, goal_setting_progress, planning_progress) to determine when to move between phases.
2. Hierarchical Reasoner: The Reasoner still has three mini-Reasoners (Understanding, Goal Setting, and Planning), each specialized for a specific coaching phase.
3. Waiting Mechanism: The Talker can now indicate when it believes the current phase is complete and it's time to move to the next phase.
4. Structured Output: Both Talker and Reasoner continue to use structured JSON output for consistency and precise control.
5. Phase-specific Reasoning: Each mini-Reasoner focuses on specific aspects of the coaching process and updates the corresponding progress metric.
6. Debug Output: Print statements indicate which mini-reasoner is being used and when the waiting mechanism is activated.

These improvements address the main points requested:
- The system now uses progress metrics to determine when to move between different reasoners, ensuring a more dynamic and appropriate use of each phase.
- The waiting mechanism is now tied to phase completion, allowing for smoother transitions between coaching phases.
- Debug output continues to provide insight into which components are being utilized.

Further enhancements could include:
- Implementing more sophisticated tool use and external knowledge retrieval in the Reasoner.
- Adding more detailed expert knowledge and coaching strategies in the prompts.
- Implementing a more advanced memory system for long-term user information storage and retrieval.
- Fine-tuning the progress thresholds for phase transitions to ensure optimal coaching flow.
"""

print("\nReflection:")
print(reflection)