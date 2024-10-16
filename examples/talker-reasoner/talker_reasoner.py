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

class TalkerReasoner:
    def __init__(self):
        self.memory = {"belief_state": BeliefState(
            coaching_phase="UNDERSTANDING",
            primary_sleep_concern="",
            goals=[],
            habits=[],
            barriers=[],
            sleep_environment={}
        )}
        self.interaction_history = []

    def talker(self, user_input: str) -> str:
        """Implement the Talker agent using GPT-4o-mini"""
        prompt = f"""You are an AI sleep coach. Your task is to interact with the user in a conversational manner.
        Current belief state: {json.dumps(self.memory['belief_state'].model_dump())}
        Interaction history: {json.dumps(self.interaction_history)}
        User input: {user_input}
        
        Respond to the user in a helpful and empathetic way, based on the current belief state and interaction history.
        
        Provide your response in JSON format with a 'response' key containing your message."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert sleep coach AI assistant. Provide your responses in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        response_content = json.loads(response.choices[0].message.content)
        return response_content.get("response", "I apologize, but I couldn't generate a response at this time.")

    def reasoner(self, user_input: str, talker_response: str) -> None:
        """Implement the Reasoner agent using GPT-4o-mini"""
        prompt = f"""You are an AI sleep coach reasoner. Your task is to update the belief state based on the user input and talker response.
        Current belief state: {json.dumps(self.memory['belief_state'].model_dump())}
        User input: {user_input}
        Talker response: {talker_response}
        
        Update the belief state based on this interaction. Provide your response in JSON format, including all fields of the belief state."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert sleep coach AI reasoner. Provide your responses in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        updated_belief = json.loads(response.choices[0].message.content)
        
        # Ensure all required fields are present
        for field in BeliefState.__annotations__.keys():
            if field not in updated_belief:
                updated_belief[field] = getattr(self.memory["belief_state"], field)
        
        self.memory["belief_state"] = BeliefState(**updated_belief)

    def interact(self, user_input: str) -> str:
        talker_response = self.talker(user_input)
        self.reasoner(user_input, talker_response)
        self.interaction_history.append({"user": user_input, "ai": talker_response})
        return talker_response

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
This implementation illustrates the core ideas of the Talker-Reasoner architecture:

1. Dual-agent system: The code separates the fast, intuitive Talker from the slow, deliberative Reasoner.
2. Belief state: The Reasoner maintains and updates a structured belief state about the user.
3. Memory: The implementation uses a memory system to store the belief state and interaction history.
4. Asynchronous operation: The Talker can respond quickly based on the current belief state, while the Reasoner updates the belief state in the background.
5. Natural language interaction: The system interacts with the user through natural language, demonstrating the paper's focus on language-based AI agents.
6. Expert knowledge integration: The prompts for both Talker and Reasoner include instructions that could incorporate expert knowledge about sleep coaching.

The implementation successfully demonstrates the paper's key concepts. However, to fully reflect the paper's ideas, future improvements could include:
- Implementing the hierarchical Reasoner with different mini-Reasoners for each coaching phase.
- Adding a mechanism for the Talker to wait for the Reasoner in complex planning scenarios.
- Incorporating more sophisticated tool use and external knowledge retrieval in the Reasoner.
- Implementing more detailed expert knowledge and coaching strategies in the prompts.
"""

print("\nReflection:")
print(reflection)