import os
import random
from typing import List, Dict, Any
import openai
import dspy
import math

# Set up OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up DSPy
dspy.settings.configure(lm=dspy.OpenAI(model="gpt-4o-mini"))

class WebElement:
    def __init__(self, element_type: str, text: str, attributes: Dict[str, str] = None):
        self.element_type = element_type
        self.text = text
        self.attributes = attributes or {}

class WebPage:
    def __init__(self, url: str, elements: List[WebElement]):
        self.url = url
        self.elements = elements

class WebEnvironment:
    def __init__(self):
        self.pages = {
            "dashboard": WebPage("https://gitlab.com/dashboard", [
                WebElement("link", "Projects"),
                WebElement("link", "Groups"),
                WebElement("link", "dotfiles"),
            ]),
            "dotfiles": WebPage("https://gitlab.com/byteblazeuser/dotfiles", [
                WebElement("link", "Project Information"),
                WebElement("link", "Repository"),
                WebElement("link", "Issues"),
                WebElement("link", "Members"),
            ]),
            "members": WebPage("https://gitlab.com/byteblazeuser/dotfiles/-/project_members", [
                WebElement("button", "Invite members"),
                WebElement("list", "Current members"),
            ]),
            "invite": WebPage("https://gitlab.com/byteblazeuser/dotfiles/-/project_members/new", [
                WebElement("textbox", "Username or email address"),
                WebElement("dropdown", "Choose a role permission"),
                WebElement("button", "Invite"),
            ]),
        }
        self.current_page = None
        self.browser_opened = False
        self.logged_in = False

    def get_observation(self) -> str:
        if not self.browser_opened:
            return "Web browser is not opened."
        if not self.logged_in:
            return "Web browser is opened but not logged in to GitLab."
        if self.current_page is None:
            return "Logged in to GitLab, but no specific page is open."
        page = self.pages[self.current_page]
        return f"Current URL: {page.url}\nElements: " + ", ".join([f"{e.element_type}: {e.text}" for e in page.elements])

    def take_action(self, action: str) -> str:
        action = action.lower()
        if "open" in action and "browser" in action:
            if not self.browser_opened:
                self.browser_opened = True
                return "Web browser opened successfully."
            else:
                return "Web browser is already open."
        if not self.browser_opened:
            return "Cannot perform action. Web browser is not opened."
        if "log in" in action and "gitlab" in action:
            if not self.logged_in:
                self.logged_in = True
                self.current_page = "dashboard"
                return "Logged in to GitLab. Now on dashboard."
            else:
                return "Already logged in to GitLab."
        if not self.logged_in:
            return "Cannot perform action. Not logged in to GitLab."
        if action.startswith("click "):
            element = action[6:]
            if element == "dotfiles" and self.current_page == "dashboard":
                self.current_page = "dotfiles"
            elif element == "Members" and self.current_page == "dotfiles":
                self.current_page = "members"
            elif element == "Invite members" and self.current_page == "members":
                self.current_page = "invite"
            elif element == "Invite" and self.current_page == "invite":
                return "Invitation sent successfully"
        elif action.startswith("type "):
            if self.current_page == "invite":
                return f"Typed '{action[5:]}' into the textbox"
        return self.get_observation()

class Planner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.plan = dspy.ChainOfThought("task -> detailed_plan")

    def forward(self, task: str) -> List[str]:
        result = self.plan(task=task)
        plan = [step.strip() for step in result.detailed_plan.split('\n') if step.strip()]
        if len(plan) < 3 or not plan[-1].endswith('.'):
            plan.append("Complete any remaining steps to fulfill the task.")
        return plan

class Controller(dspy.Module):
    def __init__(self):
        super().__init__()
        self.assess = dspy.ChainOfThought("subtask, actions, observation -> completeness, reflection")

    def forward(self, subtask: str, actions: List[str], observation: str) -> Dict[str, Any]:
        result = self.assess(
            subtask=subtask,
            actions=", ".join(actions),
            observation=observation
        )
        completeness = result.completeness.lower()
        if "complete" in completeness and self.subtask_goal_achieved(subtask, observation):
            completeness = "complete"
        elif "partial" in completeness or len(actions) > 0:
            completeness = "partial"
        else:
            completeness = "incomplete"
        return {
            "completeness": completeness,
            "reflection": result.reflection
        }

    def subtask_goal_achieved(self, subtask: str, observation: str) -> bool:
        subtask_lower = subtask.lower()
        if "open a web browser" in subtask_lower:
            return "Web browser opened" in observation
        elif "log in" in subtask_lower:
            return "Logged in to GitLab" in observation
        elif "find the 'dotfiles' repository" in subtask_lower:
            return "Current URL: https://gitlab.com/byteblazeuser/dotfiles" in observation
        elif "members page" in subtask_lower:
            return "Current URL: https://gitlab.com/byteblazeuser/dotfiles/-/project_members" in observation
        elif "invite" in subtask_lower:
            return "Invitation sent successfully" in observation
        return False

class Explorer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_action = dspy.ChainOfThought("observation, subtask, history, reflections -> action, intent")
        self.analyze_effect = dspy.ChainOfThought("previous_observation, current_observation, intent -> effect")
        self.generate_reflection = dspy.ChainOfThought("observation, subtask, action, effect -> child_reflection, sibling_reflection")

    def forward(self, observation: str, subtask: str, history: List[str], reflections: Dict[str, str]) -> Dict[str, str]:
        result = self.generate_action(
            observation=observation,
            subtask=subtask,
            history=", ".join(history),
            reflections=str(reflections)
        )
        action = result.action
        
        # Check if the action has been repeated and adjust if necessary
        if action in history:
            if "open" in action.lower() and "browser" in action.lower():
                action = "Go to the GitLab website"
            elif "log in" in action.lower():
                action = "Navigate to the GitLab dashboard"
            else:
                action = f"Try alternative action for: {action}"
        
        return {"action": action, "intent": result.intent}

    def analyze(self, previous_observation: str, current_observation: str, intent: str) -> str:
        result = self.analyze_effect(
            previous_observation=previous_observation,
            current_observation=current_observation,
            intent=intent
        )
        return result.effect

    def reflect(self, observation: str, subtask: str, action: str, effect: str) -> Dict[str, str]:
        result = self.generate_reflection(
            observation=observation,
            subtask=subtask,
            action=action,
            effect=effect
        )
        return {
            "child_reflection": result.child_reflection,
            "sibling_reflection": result.sibling_reflection
        }

class Appraiser(dspy.Module):
    def __init__(self):
        super().__init__()
        self.assess = dspy.ChainOfThought("effect, observation, subtask -> effectiveness, future_promise, reasoning")

    def forward(self, effect: str, observation: str, subtask: str) -> Dict[str, float]:
        result = self.assess(effect=effect, observation=observation, subtask=subtask)
        
        # Ensure effectiveness and future_promise are numeric
        try:
            effectiveness = float(result.effectiveness)
        except ValueError:
            effectiveness = self.interpret_score(result.effectiveness)

        try:
            future_promise = float(result.future_promise)
        except ValueError:
            future_promise = self.interpret_score(result.future_promise)

        return {
            "effectiveness": effectiveness,
            "future_promise": future_promise,
            "reasoning": result.reasoning
        }

    def interpret_score(self, assessment: str) -> float:
        assessment = assessment.lower()
        if "no" in assessment or "fail" in assessment:
            return 0.0
        elif "low" in assessment or "minor" in assessment:
            return 3.0
        elif "moderate" in assessment or "partial" in assessment:
            return 5.0
        elif "high" in assessment or "significant" in assessment:
            return 8.0
        elif "complete" in assessment or "perfect" in assessment:
            return 10.0
        else:
            return 5.0  # Default to moderate if unclear

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def add_child(self, child_state):
        child = MCTSNode(child_state, self)
        self.children.append(child)
        return child

    def update(self, reward):
        self.visits += 1
        self.value += reward

    def fully_expanded(self):
        return len(self.children) > 0

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.value / c.visits) + c_param * ((math.log(self.visits) / c.visits) ** 0.5)
            for c in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]

class MCTS:
    def __init__(self, explorer, appraiser, environment):
        self.explorer = explorer
        self.appraiser = appraiser
        self.environment = environment
        self.root = None

    def search(self, initial_state, subtask, n_iterations=100):
        self.root = MCTSNode(initial_state)

        for _ in range(n_iterations):
            node = self.select(self.root)
            child = self.expand(node, subtask)
            reward = self.simulate(child, subtask)
            self.backpropagate(child, reward)

        return self.best_action(self.root)

    def select(self, node):
        while node.fully_expanded():
            node = node.best_child()
        return node

    def expand(self, node, subtask):
        action_info = self.explorer(node.state, subtask, [], {})
        new_state = self.environment.take_action(action_info["action"])
        return node.add_child(new_state)

    def simulate(self, node, subtask):
        current_state = node.state
        depth = 0
        while depth < 5:  # Limit simulation depth
            action_info = self.explorer(current_state, subtask, [], {})
            new_state = self.environment.take_action(action_info["action"])
            effect = self.explorer.analyze(current_state, new_state, action_info["intent"])
            appraisal = self.appraiser(effect, new_state, subtask)
            if appraisal["effectiveness"] >= 8:  # Threshold for successful simulation
                return 1
            current_state = new_state
            depth += 1
        return 0

    def backpropagate(self, node, reward):
        while node is not None:
            node.update(reward)
            node = node.parent

    def best_action(self, node):
        return max(node.children, key=lambda c: c.visits).state

class WebPilot:
    def __init__(self):
        self.planner = Planner()
        self.controller = Controller()
        self.explorer = Explorer()
        self.appraiser = Appraiser()
        self.environment = WebEnvironment()
        self.mcts = MCTS(self.explorer, self.appraiser, self.environment)
        self.action_history = []
        self.max_repeated_actions = 3
        self.subtask_attempt_limit = 7

    def execute_task(self, task: str):
        task = task.replace("GitHub", "GitLab")
        subtasks = self.planner(task)
        print(f"Generated plan: {subtasks}")

        for subtask in subtasks:
            print(f"\nExecuting subtask: {subtask}")
            self.action_history.clear()
            observation = self.environment.get_observation()
            reflections = {}

            for attempt in range(self.subtask_attempt_limit):
                mcts_result = self.mcts.search(observation, subtask)
                action_info = self.explorer(mcts_result, subtask, self.action_history, reflections)
                action = action_info["action"]

                if action.lower() == "no action needed":
                    print("No action needed. Moving to next subtask.")
                    break

                print(f"Action: {action}")
                self.action_history.append(action)

                new_observation = self.environment.take_action(action)
                print(f"Observation: {new_observation}")
                
                effect = self.explorer.analyze(observation, new_observation, action_info["intent"])
                new_reflections = self.explorer.reflect(new_observation, subtask, action, effect)
                reflections.update(new_reflections)

                appraisal = self.appraiser(effect, new_observation, subtask)
                print(f"Effectiveness: {appraisal['effectiveness']}, Future Promise: {appraisal['future_promise']}")

                completion = self.controller(subtask, self.action_history, new_observation)

                if completion["completeness"] == "complete":
                    print(f"Subtask completed: {subtask}")
                    print(f"Reflection: {completion['reflection']}")
                    break

                if "already" in new_observation.lower() or appraisal['effectiveness'] >= 8:
                    print(f"Subtask seems to be completed: {subtask}")
                    break

                observation = new_observation

            if completion["completeness"] != "complete" and attempt == self.subtask_attempt_limit - 1:
                print(f"Failed to complete subtask: {subtask}")
                print(f"Reflection: {completion['reflection']}")

        print("Task execution completed.")

# Example usage
webpilot = WebPilot()
task = "Navigate to the 'Members' page of the 'dotfiles' repository on GitHub and invite 'Abishek' as a guest."
webpilot.execute_task(task)