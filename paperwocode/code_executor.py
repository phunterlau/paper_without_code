"""
Module for executing generated code in a sandbox environment.
"""
import os
import subprocess
import time
import signal
import tempfile
from typing import Dict, Any, Tuple, Optional

def create_conda_environment(env_name: str = "p2c") -> bool:
    """
    Create a conda environment for running the generated code.
    
    Args:
        env_name (str): The name of the conda environment
        
    Returns:
        bool: True if the environment was created successfully, False otherwise
    """
    try:
        # Check if the environment already exists
        result = subprocess.run(
            ["conda", "env", "list"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        
        if f"{env_name} " in result.stdout:
            print(f"Conda environment '{env_name}' already exists")
            return True
        
        # Create the environment with Python 3.10
        print(f"Creating conda environment '{env_name}'...")
        subprocess.run(
            ["conda", "create", "-n", env_name, "python=3.10", "-y"],
            check=True
        )
        
        # Install dependencies with flexible versions for better compatibility
        print(f"Installing dependencies in '{env_name}'...")
        try:
            # First try to install PyTorch packages via conda
            subprocess.run(
                ["conda", "install", "-n", env_name, "-y", "pytorch", "torchvision", "torchaudio", "-c", "pytorch"],
                check=True
            )
        except subprocess.CalledProcessError:
            print("Warning: Failed to install PyTorch packages via conda. Will try pip instead.")
        
        # Install other dependencies via pip
        subprocess.run(
            ["conda", "run", "-n", env_name, "pip", "install", 
             "numpy>=1.24.0", 
             "pandas>=2.0.0", 
             "matplotlib>=3.7.0", 
             "scikit-learn>=1.3.0", 
             "tqdm>=4.65.0", 
             "openai>=1.3.0"],
            check=True
        )
        
        # Try to install PyTorch via pip if conda installation failed
        try:
            subprocess.run(
                ["conda", "run", "-n", env_name, "pip", "install", "torch", "torchvision", "torchaudio"],
                check=True
            )
        except subprocess.CalledProcessError:
            print("Warning: Failed to install PyTorch packages via pip. Some functionality may be limited.")
        
        print(f"Conda environment '{env_name}' created successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error creating conda environment: {str(e)}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False
    
    except Exception as e:
        print(f"Error creating conda environment: {str(e)}")
        return False

def run_code_with_timeout(code_path: str, env_name: str = "p2c", timeout: int = 300) -> Tuple[bool, str, str]:
    """
    Run the generated code in the conda environment with a timeout.
    
    Args:
        code_path (str): The path to the code file
        env_name (str): The name of the conda environment
        timeout (int): The timeout in seconds
        
    Returns:
        tuple: (success, stdout, stderr)
    """
    try:
        # Create a temporary file to capture the output
        stdout_file = tempfile.NamedTemporaryFile(delete=False)
        stderr_file = tempfile.NamedTemporaryFile(delete=False)
        
        # Run the code in the conda environment
        print(f"Running code in '{env_name}' environment with {timeout}s timeout...")
        
        process = subprocess.Popen(
            ["conda", "run", "-n", env_name, "python", code_path],
            stdout=stdout_file,
            stderr=stderr_file,
            text=True
        )
        
        # Wait for the process to complete or timeout
        try:
            process.wait(timeout=timeout)
            success = process.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"Process timed out after {timeout} seconds")
            # Kill the process
            process.kill()
            success = False
        
        # Read the output
        stdout_file.close()
        stderr_file.close()
        
        with open(stdout_file.name, 'r') as f:
            stdout = f.read()
        
        with open(stderr_file.name, 'r') as f:
            stderr = f.read()
        
        # Clean up the temporary files
        os.unlink(stdout_file.name)
        os.unlink(stderr_file.name)
        
        return success, stdout, stderr
    
    except Exception as e:
        print(f"Error running code: {str(e)}")
        return False, "", str(e)

def analyze_execution_results(success: bool, stdout: str, stderr: str) -> Dict[str, Any]:
    """
    Analyze the execution results.
    
    Args:
        success (bool): Whether the execution was successful
        stdout (str): The standard output
        stderr (str): The standard error
        
    Returns:
        dict: Analysis of the execution results
    """
    analysis = {
        "success": success,
        "has_output": bool(stdout.strip()),
        "has_errors": bool(stderr.strip()),
        "error_type": None,
        "error_message": None,
        "summary": ""
    }
    
    # Analyze errors
    if stderr.strip():
        # Try to identify the type of error
        if "ImportError" in stderr or "ModuleNotFoundError" in stderr:
            analysis["error_type"] = "import_error"
            # Extract the missing module
            import_lines = [line for line in stderr.split('\n') if "ImportError" in line or "ModuleNotFoundError" in line]
            if import_lines:
                analysis["error_message"] = import_lines[0]
                analysis["summary"] += f"Missing dependency: {import_lines[0]}\n"
        
        elif "SyntaxError" in stderr:
            analysis["error_type"] = "syntax_error"
            # Extract the syntax error
            syntax_lines = [line for line in stderr.split('\n') if "SyntaxError" in line]
            if syntax_lines:
                analysis["error_message"] = syntax_lines[0]
                analysis["summary"] += f"Syntax error: {syntax_lines[0]}\n"
        
        elif "TypeError" in stderr:
            analysis["error_type"] = "type_error"
            # Extract the type error
            type_lines = [line for line in stderr.split('\n') if "TypeError" in line]
            if type_lines:
                analysis["error_message"] = type_lines[0]
                analysis["summary"] += f"Type error: {type_lines[0]}\n"
        
        elif "ValueError" in stderr:
            analysis["error_type"] = "value_error"
            # Extract the value error
            value_lines = [line for line in stderr.split('\n') if "ValueError" in line]
            if value_lines:
                analysis["error_message"] = value_lines[0]
                analysis["summary"] += f"Value error: {value_lines[0]}\n"
        
        elif "RuntimeError" in stderr:
            analysis["error_type"] = "runtime_error"
            # Extract the runtime error
            runtime_lines = [line for line in stderr.split('\n') if "RuntimeError" in line]
            if runtime_lines:
                analysis["error_message"] = runtime_lines[0]
                analysis["summary"] += f"Runtime error: {runtime_lines[0]}\n"
        
        else:
            analysis["error_type"] = "unknown_error"
            # Extract the first line of the error
            error_lines = stderr.split('\n')
            if error_lines:
                analysis["error_message"] = error_lines[0]
                analysis["summary"] += f"Unknown error: {error_lines[0]}\n"
    
    # Summarize the output
    if stdout.strip():
        # Limit the output to a reasonable length
        output_summary = stdout.strip()
        if len(output_summary) > 500:
            output_summary = output_summary[:500] + "..."
        
        analysis["summary"] += f"Output: {output_summary}\n"
    
    # Overall summary
    if success:
        analysis["summary"] += "Execution completed successfully."
    else:
        if analysis["has_errors"]:
            analysis["summary"] += "Execution failed with errors."
        else:
            analysis["summary"] += "Execution timed out or was terminated."
    
    return analysis
