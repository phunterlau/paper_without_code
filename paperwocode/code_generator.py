"""
Module for generating code based on paper summaries using Claude.
"""
import os
import json
import anthropic
import shutil
import logging
import datetime
import tiktoken
from paperwocode.code_executor import create_conda_environment, run_code_with_timeout, analyze_execution_results

# Function to estimate token count and API cost
def estimate_api_cost(prompt, response, model="claude-3-7-sonnet-20250219"):
    """
    Estimate the API cost based on token count.
    
    Args:
        prompt (str): The prompt sent to the API
        response (str): The response from the API
        model (str): The model used
        
    Returns:
        tuple: (input_tokens, output_tokens, cost_in_usd)
    """
    # Estimate token count using tiktoken
    # This is an approximation as Claude uses a different tokenizer
    enc = tiktoken.get_encoding("cl100k_base")  # Claude-compatible encoding
    
    input_tokens = len(enc.encode(prompt))
    output_tokens = len(enc.encode(response))
    
    # Claude-3-7-sonnet pricing (as of March 2025)
    # These rates may need to be updated if pricing changes
    input_cost_per_1k = 0.015  # $0.015 per 1K input tokens
    output_cost_per_1k = 0.075  # $0.075 per 1K output tokens
    
    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    total_cost = input_cost + output_cost
    
    return input_tokens, output_tokens, total_cost

# Configure logging
def setup_logger(workflow_dir):
    """
    Set up a logger for the code generation process.
    
    Args:
        workflow_dir (str): The workflow directory
        
    Returns:
        logging.Logger: The configured logger
    """
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(workflow_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Create a logger
    logger = logging.getLogger("code_generator")
    logger.setLevel(logging.INFO)
    
    # Create a file handler
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"code_generation_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    
    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(file_handler)
    
    # Add a stream handler for console output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def generate_code(markdown_content, summary, is_cs_paper, cs_subfield, output_dir="output", force=False, max_iterations=3):
    """
    Generate code based on a paper summary using Claude.
    
    Args:
        markdown_content (str): The markdown content of the paper
        summary (dict): The summary of the paper
        is_cs_paper (bool): Whether the paper is a computer science paper
        cs_subfield (str): The computer science subfield of the paper
        output_dir (str): The directory to save the generated code to
        force (bool): Whether to force code generation for non-CS papers
        max_iterations (int): Maximum number of iterations for code refinement
        
    Returns:
        str: The path to the generated code
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logger(output_dir)
    logger.info(f"Starting code generation for subfield: {cs_subfield}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Maximum iterations: {max_iterations}")
    
    # Check if this is a computer science paper
    if not is_cs_paper and not force:
        logger.warning("This does not appear to be a computer science paper.")
        logger.warning("Use the --force flag to generate code anyway.")
        print("This does not appear to be a computer science paper.")
        print("Use the --force flag to generate code anyway.")
        return None
    
    # Create the conda environment
    logger.info("Creating conda environment for code execution")
    print("\n=== Creating conda environment for code execution ===")
    env_created = create_conda_environment("p2c")
    if not env_created:
        logger.warning("Failed to create conda environment. Code execution may fail.")
        print("Warning: Failed to create conda environment. Code execution may fail.")
    
    # Select the appropriate template based on the subfield
    template_path = select_template(cs_subfield)
    
    # Initialize the Anthropic client
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    
    # Get the template content
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()
    
    # Iterative code generation and refinement
    iteration = 1
    execution_results = None
    final_code_path = None
    reflection = None
    
    while iteration <= max_iterations:
        print(f"\n=== Code Generation Iteration {iteration}/{max_iterations} ===")
        
        # Prepare the prompt based on the iteration
        if iteration == 1:
            # Initial code generation
            prompt = f"""
            You are an expert in converting research papers into working code prototypes. I'll provide you with a research paper in markdown format and its summary. Your task is to create a Python implementation that demonstrates the core idea of the paper.

            Paper Summary:
            {summary['interpretive_summary']}

            Highlights:
            {summary['highlights_explained']}

            I've selected a template for you to use as a starting point: {os.path.basename(template_path)}

            Here's what I need you to do:
            1. Analyze the paper summary and highlights to understand the core concept
            2. Create a Python implementation that demonstrates this concept
            3. Use the provided template as a starting point, but feel free to modify it as needed
            4. Include clear comments explaining how your implementation relates to the paper
            5. Make sure the code is well-structured, modular, and follows best practices
            6. Include a main function that demonstrates the implementation with a simple example
            7. Return ONLY the Python code without any explanations or markdown formatting

            The implementation should be practical and runnable, focusing on illustrating the paper's core idea rather than implementing every detail.
            
            IMPORTANT: The code will be executed in a conda environment with the following packages installed:
            - numpy (1.24.0 or newer)
            - pandas (2.0.0 or newer)
            - matplotlib (3.7.0 or newer)
            - scikit-learn (1.3.0 or newer)
            - torch (PyTorch)
            - torchvision
            - torchaudio
            - tqdm (4.65.0 or newer)
            - openai (1.3.0 or newer)
            
            Make sure your code only uses these dependencies or standard library modules.
            """
        else:
            # Code refinement based on execution results and reflection
            prompt = f"""
            You previously generated code to implement a research paper concept, but there were some issues when executing it. Please refine the code to address these issues and incorporate the feedback from the reflection.

            Paper Summary:
            {summary['interpretive_summary']}

            Highlights:
            {summary['highlights_explained']}

            Execution Results:
            Success: {execution_results['success']}
            {execution_results['summary']}

            Previous Code:
            ```python
            {previous_code}
            ```

            Reflection on Previous Implementation:
            {reflection}

            Please provide an improved version of the code that addresses the issues and incorporates the feedback from the reflection. Your implementation should:
            1. Fix any technical issues identified in the execution results
            2. Better illustrate the paper's core idea and major contributions
            3. Find the right balance of complexity (not too simple, not too complex)
            4. Include all key aspects of the paper's contributions

            Return ONLY the Python code without any explanations or markdown formatting.
            
            IMPORTANT: The code will be executed in a conda environment with the following packages installed:
            - numpy (1.24.0 or newer)
            - pandas (2.0.0 or newer)
            - matplotlib (3.7.0 or newer)
            - scikit-learn (1.3.0 or newer)
            - torch (PyTorch)
            - torchvision
            - torchaudio
            - tqdm (4.65.0 or newer)
            - openai (1.3.0 or newer)
            
            Make sure your code only uses these dependencies or standard library modules.
            """
        
        # Generate the code
        logger.info(f"Generating code using Claude (iteration {iteration}/{max_iterations})")
        print(f"Generating code using Claude (iteration {iteration})...")
        
        # Log the prompt
        logger.debug(f"Prompt for iteration {iteration}:\n{prompt}")
        
        # Call Claude API
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=8000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        generated_code = message.content[0].text
        logger.debug(f"Raw response from Claude:\n{generated_code}")
        
        # Estimate and log API cost
        input_tokens, output_tokens, cost = estimate_api_cost(prompt, generated_code)
        logger.info(f"Code generation API cost: ${cost:.4f} (Input: {input_tokens} tokens, Output: {output_tokens} tokens)")
        print(f"Estimated API cost for code generation: ${cost:.4f} (Input: {input_tokens} tokens, Output: {output_tokens} tokens)")
        
        # Clean up the generated code (remove markdown code blocks if present)
        if generated_code.startswith("```python"):
            generated_code = generated_code.split("```python", 1)[1]
        if generated_code.endswith("```"):
            generated_code = generated_code.rsplit("```", 1)[0]
        
        # Combine the template and generated code
        combined_code = f"""
# Generated code based on the paper
# Template: {os.path.basename(template_path)}
# Subfield: {cs_subfield}
# Iteration: {iteration}/{max_iterations}

{template_content}

# Implementation based on the paper
{generated_code}
"""
        
        # Save the generated code
        code_path = os.path.join(output_dir, f"generated_code_iter{iteration}.py")
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(combined_code)
        
        logger.info(f"Code generated and saved to {code_path}")
        print(f"Code generated and saved to {code_path}")
        
        # Execute the code
        logger.info(f"Executing code (iteration {iteration}/{max_iterations})")
        print(f"\n=== Executing code (iteration {iteration}) ===")
        success, stdout, stderr = run_code_with_timeout(code_path, "p2c", timeout=300)
        
        # Log the execution results
        logger.info(f"Execution {'succeeded' if success else 'failed'}")
        logger.debug(f"Standard output:\n{stdout}")
        logger.debug(f"Standard error:\n{stderr}")
        
        # Analyze the execution results
        execution_results = analyze_execution_results(success, stdout, stderr)
        logger.info(f"Analysis: {execution_results['summary']}")
        print(f"Execution {'succeeded' if success else 'failed'}")
        print(f"Analysis: {execution_results['summary']}")
        
        # Save the execution results
        results_path = os.path.join(output_dir, f"execution_results_iter{iteration}.txt")
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write(f"Success: {success}\n\n")
            f.write(f"Standard Output:\n{stdout}\n\n")
            f.write(f"Standard Error:\n{stderr}\n\n")
            f.write(f"Analysis:\n{execution_results['summary']}")
        
        logger.info(f"Execution results saved to {results_path}")
        
        # Add reflection step to evaluate the code against the paper's core idea
        print(f"\n=== Reflecting on code implementation (iteration {iteration}) ===")
        logger.info(f"Reflecting on code implementation (iteration {iteration}/{max_iterations})")
        
        reflection_prompt = f"""
        You are an expert in evaluating code implementations of research papers. I'll provide you with a paper summary and a code implementation. Your task is to evaluate how well the code illustrates the paper's core idea and major contributions.

        Paper Summary:
        {summary['interpretive_summary']}

        Highlights:
        {summary['highlights_explained']}

        Code Implementation:
        ```python
        {combined_code}
        ```

        Execution Results:
        Success: {execution_results['success']}
        {execution_results['summary']}

        Please provide a thoughtful reflection on the following:
        1. How well does the code illustrate the paper's core idea?
        2. Is the implementation too simple or too complex for the paper's concept?
        3. What aspects of the paper's contributions are well-represented in the code?
        4. What aspects are missing or could be improved?
        5. Specific recommendations for the next iteration (if needed).

        Your reflection should be concise but insightful.
        """
        
        logger.debug(f"Reflection prompt:\n{reflection_prompt}")
        
        # Call Claude API for reflection
        reflection_message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=2000,
            messages=[
                {"role": "user", "content": reflection_prompt}
            ]
        )
        
        reflection = reflection_message.content[0].text
        logger.info("Code reflection completed")
        logger.debug(f"Reflection:\n{reflection}")
        
        # Estimate and log API cost for reflection
        reflection_input_tokens, reflection_output_tokens, reflection_cost = estimate_api_cost(reflection_prompt, reflection)
        logger.info(f"Reflection API cost: ${reflection_cost:.4f} (Input: {reflection_input_tokens} tokens, Output: {reflection_output_tokens} tokens)")
        print(f"Estimated API cost for reflection: ${reflection_cost:.4f} (Input: {reflection_input_tokens} tokens, Output: {reflection_output_tokens} tokens)")
        
        # Save the reflection
        reflection_path = os.path.join(output_dir, f"code_reflection_iter{iteration}.md")
        with open(reflection_path, 'w', encoding='utf-8') as f:
            f.write(f"# Code Reflection - Iteration {iteration}\n\n")
            f.write(reflection)
        
        logger.info(f"Reflection saved to {reflection_path}")
        print(f"Reflection saved to {reflection_path}")
        
        # Print a summary of the reflection
        reflection_summary = "\n".join([line.strip() for line in reflection.split("\n") if line.strip()][:5]) + "..."
        print(f"Reflection summary:\n{reflection_summary}")
        
        # If the execution was successful, we're done
        if success and not execution_results['has_errors']:
            print(f"Code execution successful on iteration {iteration}")
            final_code_path = code_path
            break
        
        # Store the current code for the next iteration
        previous_code = combined_code
        
        # Move to the next iteration
        iteration += 1
    
    # If we've exhausted all iterations, use the last generated code
    if final_code_path is None:
        final_code_path = code_path
        logger.warning("Could not generate fully working code within the maximum number of iterations.")
        print("Warning: Could not generate fully working code within the maximum number of iterations.")
    
    # Create a copy of the final code as generated_code.py
    final_output_path = os.path.join(output_dir, "generated_code.py")
    shutil.copy2(final_code_path, final_output_path)
    
    logger.info(f"Final code saved to {final_output_path}")
    logger.info("Code generation process completed")
    print(f"Final code saved to {final_output_path}")
    
    # Save a copy of the log file with a more descriptive name
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            log_file = handler.baseFilename
            if os.path.exists(log_file):
                descriptive_log_file = os.path.join(os.path.dirname(log_file), "code_generation_complete.log")
                shutil.copy2(log_file, descriptive_log_file)
                logger.info(f"Log file copied to {descriptive_log_file}")
                break
    
    return final_output_path

def select_template(cs_subfield):
    """
    Select an appropriate template based on the computer science subfield.
    
    Args:
        cs_subfield (str): The computer science subfield
        
    Returns:
        str: The path to the selected template
    """
    # Define template mappings
    template_mappings = {
        "machine_learning": "templates/pytorch_model.py",
        "artificial_intelligence": "templates/pytorch_model.py",
        "natural_language_processing": "templates/gpt_structured_output.py",
        "computer_vision": "templates/pytorch_model.py",
        "data_science": "templates/data_analysis.py",
        "algorithms": "templates/algorithm.py",
        "systems": "templates/system.py",
        "networks": "templates/network.py",
        "security": "templates/security.py",
        "software_engineering": "templates/software.py",
        "human_computer_interaction": "templates/hci.py",
        "databases": "templates/database.py",
        "graphics": "templates/graphics.py",
        "theory": "templates/algorithm.py",
        "quantum_computing": "templates/quantum.py",
        "general_cs": "templates/general_cs.py"
    }
    
    # Get the template path
    template_path = template_mappings.get(cs_subfield, "templates/general_cs.py")
    
    # Check if the template exists
    if not os.path.exists(template_path):
        # If not, use a default template
        print(f"Template {template_path} not found. Using general_cs.py template.")
        template_path = "templates/general_cs.py"
        
        # If the default template doesn't exist, create it
        if not os.path.exists(template_path):
            os.makedirs(os.path.dirname(template_path), exist_ok=True)
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write("""
# General Computer Science Template
# This template provides a basic structure for implementing computer science concepts

def main():
    \"\"\"
    Main function to demonstrate the implementation.
    \"\"\"
    print("Implementing paper concept...")
    # Your implementation here
    
if __name__ == "__main__":
    main()
""")
    
    return template_path
