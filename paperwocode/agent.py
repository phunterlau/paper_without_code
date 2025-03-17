"""
Module for orchestrating the workflow of the application.
"""
import os
import sys
from paperwocode.pdf_downloader import download_pdf, extract_arxiv_id
from paperwocode.pdf_to_markdown import convert_pdf_to_markdown
from paperwocode.paper_summarizer import summarize_paper
from paperwocode.code_generator import generate_code

def run_agent_workflow(arxiv_url, force=False, output_dir="output", max_iterations=3, no_cache=False):
    """
    Run the agent workflow.
    
    Args:
        arxiv_url (str): The arXiv URL
        force (bool): Whether to force code generation for non-CS papers
        output_dir (str): The base directory to save the output to
        max_iterations (int): Maximum number of iterations for code refinement
        no_cache (bool): Whether to skip using cached results
        
    Returns:
        None
    """
    try:
        # Step 1: Download the PDF
        print("\n=== Step 1: Downloading PDF ===")
        # Extract arXiv ID from URL to check for cached results
        arxiv_id = None
        workflow_dir = None
        pdf_path = None
        
        try:
            arxiv_id = extract_arxiv_id(arxiv_url)
            clean_id = arxiv_id.replace('/', '_')
            workflow_dir = os.path.join(output_dir, "workflows", clean_id)
            pdf_path = os.path.join(workflow_dir, f"{clean_id}.pdf")
            
            # Check if PDF already exists
            if os.path.exists(pdf_path) and not no_cache:
                print(f"Using cached PDF: {pdf_path}")
            else:
                # Download the PDF
                pdf_path, arxiv_id, workflow_dir = download_pdf(arxiv_url, output_dir)
        except:
            # If extraction fails, download the PDF
            pdf_path, arxiv_id, workflow_dir = download_pdf(arxiv_url, output_dir)
        
        if not pdf_path:
            print("Failed to download PDF.")
            sys.exit(1)
        
        print(f"Using workflow directory: {workflow_dir}")
        
        # Step 2: Convert the PDF to markdown
        print("\n=== Step 2: Converting PDF to markdown ===")
        markdown_path = os.path.join(workflow_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}.md")
        
        # Check if markdown already exists
        if os.path.exists(markdown_path) and not no_cache:
            print(f"Using cached markdown: {markdown_path}")
            with open(markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
        else:
            # Convert the PDF to markdown
            markdown_path, markdown_content = convert_pdf_to_markdown(pdf_path, workflow_dir)
        
        if not markdown_path or not markdown_content:
            print("Failed to convert PDF to markdown.")
            sys.exit(1)
        
        # Step 3: Summarize the paper
        print("\n=== Step 3: Summarizing paper ===")
        summary_path = os.path.join(workflow_dir, f"{os.path.splitext(os.path.basename(markdown_path))[0]}-summary.md")
        
        # Check if summary already exists
        if os.path.exists(summary_path) and not no_cache:
            print(f"Using cached summary: {summary_path}")
            # Load the summary from the file
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary_content = f.read()
            
            # Extract is_cs_paper and cs_subfield from the summary
            is_cs_paper = "computer science" in summary_content.lower()
            cs_subfield = "general_cs"  # Default
            
            # Try to determine the subfield from the summary
            subfields = [
                "machine_learning", "artificial_intelligence", "natural_language_processing",
                "computer_vision", "data_science", "algorithms", "systems", "networks",
                "security", "software_engineering", "human_computer_interaction",
                "databases", "graphics", "theory", "quantum_computing"
            ]
            
            for subfield in subfields:
                if subfield.replace("_", " ") in summary_content.lower():
                    cs_subfield = subfield
                    break
            
            # Create a minimal summary dictionary
            summary = {
                "interpretive_summary": summary_content,
                "highlights_explained": summary_content,
                "first_pass": "",
                "second_pass": "",
                "third_pass": ""
            }
        else:
            # Summarize the paper
            summary, summary_path, is_cs_paper, cs_subfield = summarize_paper(markdown_path, markdown_content, workflow_dir)
        
        if not summary:
            print("Failed to summarize paper.")
            sys.exit(1)
        
        # Step 4: Check if this is a computer science paper
        print("\n=== Step 4: Checking if this is a computer science paper ===")
        if not is_cs_paper and not force:
            print("This does not appear to be a computer science paper.")
            print("Use the --force flag to generate code anyway.")
            user_input = input("Do you want to proceed anyway? (y/n): ")
            if user_input.lower() != 'y':
                print("Exiting.")
                sys.exit(0)
        
        # Step 5: Generate code
        print("\n=== Step 5: Generating code ===")
        final_code_path = os.path.join(workflow_dir, "generated_code.py")
        
        # Check if code already exists
        if os.path.exists(final_code_path) and not no_cache:
            print(f"Using cached code: {final_code_path}")
            code_path = final_code_path
        else:
            # Generate code
            code_path = generate_code(
                markdown_content, 
                summary, 
                is_cs_paper, 
                cs_subfield, 
                workflow_dir, 
                force,
                max_iterations
            )
        
        if not code_path:
            print("Failed to generate code.")
            sys.exit(1)
        
        # Step 6: Done
        print("\n=== Done! ===")
        print(f"Workflow directory: {workflow_dir}")
        print(f"PDF downloaded to: {pdf_path}")
        print(f"Markdown saved to: {markdown_path}")
        print(f"Summary saved to: {summary_path}")
        print(f"Code generated at: {code_path}")
        print("\nYou can run the generated code with:")
        print(f"python {code_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)
