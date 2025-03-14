# PaperWithoutCode

An agentic tool to generate code prototypes from research papers in <https://paperwithoutcode.com/>. No Llama-index, no LangChain, just get back to the root of the idea.

![meme](images/90j0pl.jpg)

## Overview

PaperWithoutCode is a Python project of <https://paperwithoutcode.com/> that takes an arXiv article PDF link as input, downloads the PDF, converts it to markdown, generates a summary of the paper, create a Python script that demonstrates the paper's core idea, execute in a sandbox, review and improve. It leverages an agentic workflow to dynamically manage all the tasks.

The generated code is executed in a sandbox conda environment, and the results are analyzed to refine the code if necessary. This process is repeated up to a specified number of iterations to produce a working prototype that best illustrate the paper's idea.

## Features

- Download PDFs from arXiv links
- Convert PDFs to markdown using the `markitdown` package
- Generate comprehensive paper summaries using a three-pass approach
- Select an appropriate template based on the paper's subfield
- Generate code using Claude Sonnet 3.7
- Execute the generated code in a sandbox conda environment
- Analyze execution results and refine the code iteratively
- Support for various computer science subfields with specialized templates

## Installation

### Automatic Setup (Recommended)

#### On Linux/macOS:
```bash
git clone https://github.com/yourusername/paperwocode.git
cd paperwocode
chmod +x setup.sh  # Make the setup script executable
./setup.sh
```

#### On Windows:
```cmd
git clone https://github.com/yourusername/paperwocode.git
cd paperwocode
setup.bat
```

The setup script will:
1. Create the conda environment with all required dependencies
2. Prompt you for API keys if they're not already set
3. Create the necessary output directories
4. Provide instructions for running the project

### Manual Setup

1. Clone the repository:
   ```
   git clone https://github.com/phunterlau/paper_without_code.git
   cd paperwocode
   ```

2. Create and activate the conda environment:
   ```
   conda env create -f environment.yml
   conda activate paperwocode
   ```

   This will create a conda environment with all the required dependencies with compatible versions.

3. Set the required API keys as environment variables:
   ```
   export ANTHROPIC_API_KEY=your_anthropic_api_key
   export OPENAI_API_KEY=your_openai_api_key
   ```

   On Windows, use:
   ```
   set ANTHROPIC_API_KEY=your_anthropic_api_key
   set OPENAI_API_KEY=your_openai_api_key
   ```

4. Create the output directory structure:
   ```
   mkdir -p output/workflows
   ```

## Usage

Run the script with an arXiv URL:

```
python main.py https://arxiv.org/abs/2411.16905
```

Or use the default URL (https://arxiv.org/abs/2411.16905):

```
python main.py
```

### Command-line Arguments

- `arxiv_url`: URL to the arXiv paper (default: https://arxiv.org/abs/2411.16905)
- `--force`: Force code generation even for non-CS papers
- `--output`, `-o`: Output directory for generated files (default: "output")
- `--iterations`, `-i`: Maximum number of iterations for code refinement (default: 3)
- `--no-cache`: Skip using cached results and rerun all steps

## Project Structure

```
.
├── main.py                         # Main entry point
├── setup.py                        # Package setup script
├── setup.sh                        # Setup script for Linux/macOS
├── setup.bat                       # Setup script for Windows
├── requirements.txt                # Project dependencies
├── environment.yml                 # Conda environment specification
├── paperwocode/                    # Main package directory
│   ├── __init__.py                 # Package initialization
│   ├── pdf_downloader.py           # Module for downloading PDFs from arXiv
│   ├── pdf_to_markdown.py          # Module for converting PDFs to markdown
│   ├── paper_summarizer.py         # Module for summarizing papers
│   ├── code_generator.py           # Module for generating code
│   ├── code_executor.py            # Module for executing code in a sandbox
│   └── agent.py                    # Module for orchestrating the workflow
├── templates/                      # Templates for different types of papers
│   ├── gpt_structured_output.py    # Template for LLM-related papers
│   ├── pytorch_model.py            # Template for model-related papers
│   ├── data_analysis.py            # Template for data science papers
│   ├── algorithm.py                # Template for algorithm-related papers
│   └── general_cs.py               # Template for general CS papers
└── output/                         # Base output directory
    └── workflows/                  # Workflows directory
        └── {arxiv_id}/             # Workflow directory for each paper
            ├── {arxiv_id}.pdf      # Downloaded PDF
            ├── {arxiv_id}.md       # Converted markdown
            ├── {arxiv_id}-summary.md # Paper summary
            ├── {arxiv_id}-markmap.md # Mind map
            ├── generated_code.py   # Final generated code
            ├── generated_code_iter1.py # First iteration of code
            ├── generated_code_iter2.py # Second iteration of code
            ├── generated_code_iter3.py # Third iteration of code
            ├── execution_results_iter1.txt # First execution results
            ├── execution_results_iter2.txt # Second execution results
            ├── execution_results_iter3.txt # Third execution results
            ├── code_reflection_iter1.md # First code reflection
            ├── code_reflection_iter2.md # Second code reflection
            ├── code_reflection_iter3.md # Third code reflection
            └── logs/                # Logs directory
                ├── code_generation_*.log # Timestamped log files
                └── code_generation_complete.log # Final log file
```

## Workflow

1. **Download PDF**: The PDF is downloaded from the provided arXiv URL.
2. **Convert to Markdown**: The PDF is converted to markdown using the `markitdown` package.
3. **Summarize Paper**: The paper is summarized using a three-pass approach.
4. **Check if CS Paper**: The system determines if the paper is a computer science paper and identifies its subfield.
5. **Generate Code**: Claude Sonnet is used to generate a Python script based on the paper's core idea.
6. **Execute Code**: The generated code is executed in a sandbox conda environment.
7. **Analyze Results**: The execution results are analyzed to identify any issues.
8. **Refine Code**: If issues are found, the code is refined and the process is repeated.
9. **Final Output**: The final code is saved to the output directory.

## Caching System

The project includes a caching system that saves time by reusing previously generated results:

1. **Automatic Caching**: Each step of the workflow checks if the output already exists in the workflow directory.
2. **Cache Usage**: If a cached result is found, it's used instead of rerunning the step, which can save significant time, especially for the summarization and code generation steps.
3. **Cache Invalidation**: Use the `--no-cache` flag to skip using cached results and rerun all steps.
4. **Workflow Directory**: All cached files are stored in a dedicated directory for each paper (`output/workflows/{arxiv_id}/`), making it easy to manage and share results.

This caching system is particularly useful when:
- You want to experiment with different code generation parameters without re-processing the paper
- You're working with the same paper multiple times
- You want to share the results with others

## Cost Estimation System

The project includes a cost estimation system that calculates and displays the estimated API costs for each step:

1. **Token Counting**: The system counts input and output tokens for each API call using the tiktoken library.
2. **Cost Calculation**: Based on the token counts, the system calculates the estimated cost using Claude's pricing model.
3. **Transparent Reporting**: The estimated cost is displayed after each API call, showing:
   - Total cost in USD
   - Input token count and cost
   - Output token count and cost
4. **Logging**: All cost estimates are logged for future reference and budget planning.

This cost estimation system helps you:
- Monitor API usage and costs in real-time
- Plan and budget for larger projects
- Optimize prompts to reduce costs
- Track expenses across different papers

## Code Reflection System

The project includes an intelligent reflection system that evaluates how well the generated code illustrates the paper's core idea:

1. **Automated Evaluation**: After each code generation and execution step, Claude Sonnet evaluates the code against the paper's core idea and major contributions.
2. **Comprehensive Assessment**: The reflection considers:
   - How well the code illustrates the paper's core idea
   - Whether the implementation is too simple or too complex
   - Which aspects of the paper's contributions are well-represented
   - Which aspects are missing or could be improved
   - Specific recommendations for the next iteration
3. **Feedback Loop**: The reflection is used as input for the next iteration of code generation, creating a continuous improvement cycle.
4. **Documentation**: All reflections are saved as markdown files in the workflow directory for future reference.

This reflection system ensures that the generated code not only runs correctly but also effectively demonstrates the paper's key concepts and contributions.

## Logging System

The project includes a comprehensive logging system that records all steps of the code generation process:

1. **Detailed Logs**: All code generation steps, API calls, execution results, analysis, and reflections are logged in detail.
2. **Log Directory**: Logs are stored in a dedicated directory for each paper (`output/workflows/{arxiv_id}/logs/`).
3. **Timestamped Logs**: Each log file is timestamped to track different runs.
4. **Complete Record**: The entire process is recorded, including:
   - Claude API prompts and responses
   - Code execution outputs and errors
   - Analysis of execution results
   - Code reflections and evaluations
   - Final code generation results

This logging system is particularly useful for:
- Debugging code generation issues
- Understanding why certain code was generated
- Tracking the evolution of code across iterations
- Auditing the code generation process

## Templates

The system includes several templates for different types of computer science papers:

- **gpt_structured_output.py**: For LLM-related papers, using GPT-4o-mini for structured output.
- **pytorch_model.py**: For model-related papers, using PyTorch.
- **data_analysis.py**: For data science papers, using pandas, scikit-learn, and matplotlib.
- **algorithm.py**: For algorithm-related papers, with benchmarking capabilities.
- **general_cs.py**: For general computer science papers.

## Requirements

- Python 3.10+
- Conda (for creating the sandbox environment)
- Anthropic API key (for Claude Sonnet)
- OpenAI API key (for GPT-4o-mini)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
