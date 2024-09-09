# Code for "SELF-DISCOVER: Large Language Models Self-Compose Reasoning Structures"

Blog post <https://paperwithoutcode.com/self-discover-large-language-models-self-compose-reasoning-structures/>

The paper "SELF-DISCOVER: Large Language Models Self-Compose Reasoning Structures" introduces an innovative framework that significantly enhances the reasoning capabilities of Large Language Models (LLMs) by enabling them to autonomously discover and compose task-specific reasoning structures. By leveraging multiple atomic reasoning modules such as critical thinking and step-by-step analysis, SELF-DISCOVER dynamically adapts to the unique intrinsic reasoning needs of each task. This framework demonstrates substantial improvements on benchmarks like BigBench-Hard and MATH, outperforming traditional methods such as Chain of Thought (CoT) and CoT-Self-Consistency, while requiring significantly less computational cost. One of the most compelling aspects of SELF-DISCOVER is its universal applicability; reasoning structures discovered by one LLM have been successfully transferred to other models, showcasing robust generalizability across different LLM families. This novel approach not only pushes the boundaries of LLM performance but also introduces a more efficient and scalable method for reasoning in AI applications. The framework's potential impact is profound, paving the way for better agents and enhanced LLM applications in diverse fields such as legal reasoning, medical diagnosis, and advanced scientific research. Future research could explore error mitigation, broader benchmarks, and collaborative prompting techniques, further unlocking the capabilities of LLMs in complex problem-solving.

Example questions:

* "Calculate the area of a triangle with base 6 cm and height 8 cm.",
* "Explain why the sky appears blue."
* "Solve the equation: 2x + 5 = 13"
* "A rectangular garden is 3 meters longer than it is wide. If the perimeter of the garden is 26 meters, what are its dimensions?"
* "In a group of 5 friends, if Alice is taller than Bob, Bob is taller than Charlie, Charlie is shorter than David, and David is shorter than Eve, who is the tallest and who is the shortest?"
* "A ball is thrown vertically upward with an initial velocity of 20 m/s from a height of 1.5 m above the ground. How long does it take for the ball to reach its maximum height, and what is that height? (Assume g = 9.8 m/s^2)"
* "Design a system to automatically water plants in a garden based on soil moisture levels and weather forecasts."
* "A self-driving car is about to crash. It can either swerve left and hit a group of elderly pedestrians, or swerve right and hit a group of young children. What ethical considerations should guide its decision?"
* "Explain the process of photosynthesis and its importance for life on Earth."
* "Analyze the causes and consequences of the Industrial Revolution.",
* "Write a haiku about the changing seasons."
* "Estimate the number of piano tuners in Chicago."
* "Describe how machine learning algorithms can be used to detect fraud in financial transactions."