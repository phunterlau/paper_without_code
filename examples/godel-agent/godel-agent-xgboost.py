import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from openai import OpenAI
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# Initialize OpenAI client for LLM interactions
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Define base models for the Gödel Agent framework
class Action(BaseModel):
    name: str
    code: Optional[str] = ""
    reasoning: Optional[str] = ""

class GodelAgent(BaseModel):
    policy: str
    learning_algorithm: str

class Environment(BaseModel):
    state: str
    feedback: float = 0.0

# XGBoostAutoMLAgent: Implements the Gödel Agent for XGBoost AutoML
class XGBoostAutoMLAgent(GodelAgent):
    best_params: Dict[str, Any] = Field(default_factory=dict)
    best_score: float = Field(default=float('inf'))
    feature_importance: Dict[str, float] = Field(default_factory=dict)
    selected_features: List[str] = Field(default_factory=list)
    engineered_feature_names: List[str] = Field(default_factory=list)

    def __init__(self, policy: str, learning_algorithm: str):
        super().__init__(policy=policy, learning_algorithm=learning_algorithm)
        # Initialize with default XGBoost parameters
        self.best_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'min_child_weight': 1,
            'subsample': 1,
            'colsample_bytree': 1
        }

    # Method for LLM-driven feature engineering
    def engineer_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        print(f"Starting feature engineering. Input shapes - X: {X.shape}, y: {y.shape}")
        
        # Prepare data summaries for LLM prompt
        feature_info = X.describe().to_dict()
        target_info = y.describe().to_dict()

        # Construct prompt for LLM to suggest feature engineering operations
        prompt = f"""
        As an AI specializing in feature engineering for machine learning, your task is to design advanced features for a regression problem. You have access to the following data:

        Features: {json.dumps(feature_info, indent=2)}
        Target: {json.dumps(target_info, indent=2)}

        Based on this information, suggest a list of feature engineering operations. Each operation should be one of the following types:
        1. Mathematical transformation (e.g., log, square, cube root)
        2. Interaction between features (e.g., multiplication, division)
        3. Binning of continuous variables
        4. Aggregation of features
        5. Domain-specific transformations

        Provide your suggestions in a structured JSON format with the following schema:
        {{
            "operations": [
                {{
                    "type": "string (one of the 5 types mentioned above)",
                    "description": "string (brief description of the operation)",
                    "features_involved": ["list of feature names"],
                    "operation": "string (name of the operation, e.g., 'log', 'square', 'multiply', 'divide', 'bin')"
                }}
            ]
        }}

        Aim for a diverse set of operations that could potentially improve the model's performance.
        Ensure that all feature names used exist in the provided feature set.
        """

        # Get feature engineering suggestions from LLM
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI specializing in feature engineering for machine learning."},
                    {"role": "user", "content": prompt}
                ],
                response_format={ "type": "json_object" }
            )

            operations = json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error in getting or parsing LLM response: {str(e)}")
            return X, []

        # Apply suggested feature engineering operations
        new_features = X.copy()
        new_feature_names = []
        for op in operations['operations']:
            try:
                new_feature_name = f"{op['type']}_{'-'.join(op['features_involved'])}"
                features = op['features_involved']
                
                # Validate features
                if not all(feature in X.columns for feature in features):
                    print(f"Skipping operation due to missing features: {op['description']}")
                    continue
                
                # Apply the operation based on its type
                if op['operation'] == 'log':
                    new_features[new_feature_name] = np.log1p(X[features[0]])
                elif op['operation'] == 'square':
                    new_features[new_feature_name] = X[features[0]] ** 2
                elif op['operation'] == 'multiply':
                    new_features[new_feature_name] = X[features[0]] * X[features[1]]
                elif op['operation'] == 'divide':
                    new_features[new_feature_name] = X[features[0]] / (X[features[1]] + 1e-8)  # Avoid division by zero
                elif op['operation'] == 'bin':
                    new_features[new_feature_name] = pd.qcut(X[features[0]], q=5, labels=False, duplicates='drop')
                else:
                    print(f"Unknown operation: {op['operation']}")
                    continue

                new_feature_names.append(new_feature_name)
                print(f"Successfully applied operation: {op['description']}")
            except Exception as e:
                print(f"Error applying operation {op['description']}: {str(e)}")

        self.engineered_feature_names = new_feature_names
        print(f"Feature engineering completed. Output shapes - X: {new_features.shape}, New features: {len(new_feature_names)}")
        return new_features, new_feature_names

    # Method to train and evaluate the XGBoost model
    def train_and_evaluate(self, X, y):
        print(f"Initial shapes - X: {X.shape}, y: {y.shape}")
        
        # Apply feature engineering if there are engineered features
        if self.engineered_feature_names:
            X, new_feature_names = self.engineer_features(X, y)
            print(f"After engineering - X: {X.shape}, y: {y.shape}")

        # Apply feature selection if there are selected features
        if self.selected_features:
            available_features = set(X.columns) & set(self.selected_features)
            X = X[list(available_features)]
            print(f"After feature selection - X: {X.shape}, y: {y.shape}")

        # Ensure X and y have the same number of samples
        if len(X) != len(y):
            print(f"Mismatch in samples. X: {len(X)}, y: {len(y)}")
            min_samples = min(len(X), len(y))
            X = X.iloc[:min_samples]
            y = y.iloc[:min_samples]
            print(f"Adjusted to min samples. New shapes - X: {X.shape}, y: {y.shape}")

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        # Train XGBoost model with current best parameters
        model = XGBRegressor(**self.best_params, random_state=42)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate model performance
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Update feature importance
        self.feature_importance = dict(zip(X.columns, model.feature_importances_))
        return rmse, r2

# AutoMLGodelAgentFramework: Implements the main loop for the Gödel Agent
class AutoMLGodelAgentFramework:
    def __init__(self, initial_policy: str, initial_learning_algorithm: str):
        self.agent = XGBoostAutoMLAgent(policy=initial_policy, learning_algorithm=initial_learning_algorithm)
        self.environment = Environment(state="")

    # Main evolution loop for the Gödel Agent
    def evolve(self, X: pd.DataFrame, y: pd.Series, max_iterations: int = 10):
        task_output = f"Task: AutoML for XGBoost Regression on California Housing Dataset\n\n"
        for i in range(max_iterations):
            print(f"\n--- Iteration {i+1}/{max_iterations} ---")
            task_output += f"\n--- Iteration {i+1}/{max_iterations} ---\n"
            print(f"Current policy: {self.agent.policy}")
            print(f"Current learning algorithm: {self.agent.learning_algorithm}")
            task_output += f"Current policy: {self.agent.policy}\n"
            task_output += f"Current learning algorithm: {self.agent.learning_algorithm}\n\n"
            
            # Decide on actions to take
            print("\nDeciding actions...")
            actions = self._decide_actions(X, y)
            task_output += "Decided actions:\n"
            for action in actions:
                print(f"- {action.name}")
                task_output += f"- {action.name}\n"
            task_output += "\n"
            
            # Execute decided actions
            print("\nExecuting actions:")
            for action in actions:
                print(f"\nExecuting: {action.name}")
                action_output, X = self._execute_action(action, X, y)
                print(action_output)
                task_output += action_output + "\n"
            
            # Train and evaluate the model
            print("\nTraining and evaluating model...")
            rmse, r2 = self.agent.train_and_evaluate(X, y)
            
            # Report current performance and model details
            print(f"Current performance: RMSE = {rmse:.4f}, R2 = {r2:.4f}")
            task_output += f"Current performance: RMSE = {rmse:.4f}, R2 = {r2:.4f}\n"
            print(f"Best parameters: {self.agent.best_params}")
            task_output += f"Best parameters: {self.agent.best_params}\n"
            print("Top 5 important features:")
            for feature, importance in sorted(self.agent.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {feature}: {importance:.4f}")
            task_output += f"Top 5 important features: {dict(sorted(self.agent.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])}\n\n"
            
            # Check for new best score
            if rmse < self.agent.best_score:
                self.agent.best_score = rmse
                print(f"\nNew best score achieved! RMSE: {rmse:.4f}")
                task_output += f"New best score achieved! RMSE: {rmse:.4f}\n\n"
            
            # Check termination conditions
            if i == max_iterations - 1 or r2 > 0.95:
                print("\nAutoML process completed.")
                task_output += "AutoML process completed.\n"
                break
        
        return task_output

    # Method to decide on actions using LLM
    def _decide_actions(self, X, y) -> List[Action]:
        feature_info = self.agent.feature_importance
        top_features = dict(sorted(feature_info.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Construct prompt for LLM to decide on actions
        prompt = f"""
        You are an AutoML system for XGBoost models working on the California Housing dataset. Your current policy is:
        {self.agent.policy}
        
        Your current learning algorithm is:
        {self.agent.learning_algorithm}
        
        Current best parameters: {self.agent.best_params}
        Current best score (RMSE): {self.agent.best_score}
        Top 5 important features: {top_features}
        
        Decide on a sequence of actions to improve the model's performance. Available actions are:
        1. update_hyperparameters: Modify XGBoost hyperparameters
        2. feature_selection: Select or engineer features
        3. feature_engineering: Apply advanced feature engineering techniques
        4. update_policy: Modify the current policy
        5. update_learning_algorithm: Modify the current learning algorithm
        
        Return a list of actions in JSON format, including reasoning for each action. Each action should have a 'name' field matching one of the above actions, and optionally 'code' and 'reasoning' fields.
        If 'code' is provided for 'update_hyperparameters', ensure it's a string representation of a JSON object.
        """
        
        # Get action decisions from LLM
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert in AutoML and XGBoost optimization for regression tasks."},
                {"role": "user", "content": prompt}
            ],
            response_format={ "type": "json_object" }
        )
        
        actions_json = json.loads(response.choices[0].message.content)
        
        # Parse and validate actions from LLM response
        actions = []
        if "actions" in actions_json:
            for action in actions_json["actions"]:
                try:
                    if isinstance(action, dict):
                        if "name" not in action:
                            action["name"] = action.get("action", "unknown_action")
                        if "code" in action and isinstance(action["code"], dict):
                            action["code"] = json.dumps(action["code"])
                        actions.append(Action(**action))
                    else:
                        print(f"Skipping invalid action: {action}")
                except Exception as e:
                    print(f"Error creating Action object: {e}")
        else:
            print("No 'actions' key found in LLM response. Using entire response as a single action.")
            try:
                actions.append(Action(name="composite_action", code=json.dumps(actions_json), reasoning="Composite action from LLM response"))
            except Exception as e:
                print(f"Error creating composite Action object: {e}")
        
        if not actions:
            print("No valid actions found. Using default action.")
            actions.append(Action(name="default_action", reasoning="No valid actions provided by LLM"))
        
        return actions

# Method to execute decided actions
    def _execute_action(self, action: Action, X: pd.DataFrame, y: pd.Series) -> Tuple[str, pd.DataFrame]:
        output = f"Executing action: {action.name}\n"
        if action.reasoning:
            output += f"Reasoning: {action.reasoning}\n"
        
        # Execute action based on its name
        if action.name == "update_hyperparameters":
            if action.code:
                try:
                    # Parse and update hyperparameters
                    new_params = json.loads(action.code)
                    self.agent.best_params.update(new_params)
                    output += f"Updated hyperparameters: {new_params}\n"
                except json.JSONDecodeError:
                    output += "Failed to parse hyperparameters. No updates applied.\n"
            else:
                output += "No hyperparameter updates provided.\n"
        elif action.name == "feature_selection":
            if action.code:
                try:
                    # Parse and apply feature selection
                    selected_features = json.loads(action.code)
                    if isinstance(selected_features, list) and all(isinstance(f, str) for f in selected_features):
                        available_features = set(X.columns) & set(selected_features)
                        self.agent.selected_features = list(available_features)
                        X = X[self.agent.selected_features]
                        output += f"Updated selected features: {self.agent.selected_features}\n"
                    else:
                        output += "Invalid feature selection format. Expected a list of feature names.\n"
                except json.JSONDecodeError:
                    output += "Failed to parse selected features. No feature selection performed.\n"
            else:
                output += "No feature selection performed.\n"
        elif action.name == "feature_engineering":
            # Apply feature engineering
            X_engineered, new_features = self.agent.engineer_features(X, y)
            output += f"Performed feature engineering. New features: {new_features}\n"
            X = X_engineered  # Update X with engineered features
        elif action.name == "update_policy":
            if action.code:
                # Update agent's policy
                self.agent.policy = action.code
                output += f"Updated policy: {self.agent.policy}\n"
            else:
                output += "No policy updates provided.\n"
        elif action.name == "update_learning_algorithm":
            if action.code:
                # Update agent's learning algorithm
                self.agent.learning_algorithm = action.code
                output += f"Updated learning algorithm: {self.agent.learning_algorithm}\n"
            else:
                output += "No learning algorithm updates provided.\n"
        else:
            output += f"Unknown action: {action.name}\n"
        
        return output, X

# Main function to run the AutoML process
def main():
    # Load the California Housing dataset
    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target)

    # Define initial policy and learning algorithm
    initial_policy = "Start with default XGBoost parameters and gradually refine based on performance and feature engineering"
    initial_learning_algorithm = "Use performance feedback to guide hyperparameter tuning, feature selection, and LLM-driven feature engineering"
    
    # Initialize the AutoML Gödel Agent
    automl_agent = AutoMLGodelAgentFramework(initial_policy, initial_learning_algorithm)
    
    print("AutoML XGBoost Optimization for California Housing Dataset")
    # Run the evolution process
    task_output = automl_agent.evolve(X, y)
    print("\nFinal AutoML Output:")
    print(task_output)

    # Plot feature importance
    feature_importance = automl_agent.agent.feature_importance
    sorted_idx = np.argsort(list(feature_importance.values()))
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(sorted_idx)), np.array(list(feature_importance.values()))[sorted_idx])
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels(np.array(list(feature_importance.keys()))[sorted_idx])
    ax.set_title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

"""
Key Steps and Rationales:

1. Gödel Agent Framework:
   - Implements recursive self-improvement through the AutoMLGodelAgentFramework class.
   - Rationale: Enables the agent to modify its own logic and behavior, exploring the full design space.

2. XGBoost AutoML Agent:
   - Encapsulates XGBoost-specific logic in the XGBoostAutoMLAgent class.
   - Rationale: Provides a flexible structure for managing model parameters and feature engineering.

3. LLM-Driven Decision Making:
   - Uses LLMs to decide on actions and suggest feature engineering operations.
   - Rationale: Leverages advanced AI capabilities for intelligent decision-making and knowledge incorporation.

4. Feature Engineering:
   - Implements LLM-guided feature engineering in the engineer_features method.
   - Rationale: Allows for creative and context-aware feature creation beyond predefined rules.

5. Recursive Self-Improvement Loop:
   - Implemented in the evolve method, continuously refining the model and agent behavior.
   - Rationale: Core mechanism for achieving progressive improvements over time.

6. Flexible Action Execution:
   - Supports various action types including hyperparameter tuning and policy updates.
   - Rationale: Enables diverse optimization strategies and adaptation to different problem domains.

7. Performance Tracking:
   - Maintains best score and updates when improvements are made.
   - Rationale: Ensures the agent is always progressing towards better solutions.

8. Visualization:
   - Includes feature importance visualization for model interpretability.
   - Rationale: Aids in understanding the model's decision-making process and validating improvements.

Future Improvements:

1. Enhanced Optimization Modules:
   - Implement more sophisticated hyperparameter tuning algorithms (e.g., Bayesian optimization).
   - Rationale: Could lead to faster convergence and better final performance.

2. Meta-Learning Capabilities:
   - Incorporate transfer learning to leverage knowledge from previous runs or similar datasets.
   - Rationale: Could improve efficiency and performance on new, related tasks.

3. Multi-Model Support:
   - Extend the framework to support multiple model types beyond XGBoost.
   - Rationale: Increases versatility and applicability to diverse machine learning tasks.

4. Adaptive Stopping Criteria:
   - Implement more sophisticated stopping criteria based on performance plateaus or time constraints.
   - Rationale: Optimizes the trade-off between computation time and model improvement.

5. Explainable AI Integration:
   - Incorporate methods for explaining the agent's decisions and model predictions.
   - Rationale: Enhances trust and interpretability, crucial for real-world applications.

6. Collective Intelligence:
   - Explore interactions between multiple Gödel Agents for collaborative problem-solving.
   - Rationale: Could lead to more robust and diverse solution strategies.

7. Safety Measures:
   - Implement constraints to prevent harmful self-modifications or actions.
   - Rationale: Ensures safe and controlled operation as the agent becomes more autonomous.

8. Dynamic Feature Importance:
   - Implement methods to track feature importance changes over iterations.
   - Rationale: Provides insights into the evolving understanding of the problem by the agent.

This implementation serves as a proof-of-concept for the Gödel Agent approach in AutoML,
demonstrating the potential of self-referential, LLM-driven agents in optimizing machine
learning models. Future work can build upon this foundation to create more sophisticated,
efficient, and widely applicable AutoML systems.
"""