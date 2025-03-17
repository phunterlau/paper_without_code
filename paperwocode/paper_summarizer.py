"""
Module for summarizing papers.
"""
import os
import json
import anthropic

def summarize_paper(markdown_path, markdown_content, output_dir):
    """
    Summarize a paper.
    
    Args:
        markdown_path (str): The path to the markdown file
        markdown_content (str): The content of the markdown file
        output_dir (str): The directory to save the summary to
        
    Returns:
        tuple: (summary, summary_path, is_cs_paper, cs_subfield)
    """
    try:
        # Define the summary path
        summary_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(markdown_path))[0]}-summary.md")
        
        # Check if the summary already exists
        if os.path.exists(summary_path):
            print(f"Summary already exists at {summary_path}")
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
            
            return summary, summary_path, is_cs_paper, cs_subfield
        
        # Initialize the Anthropic client
        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
        )
        
        # Prepare the prompt
        prompt = f"""
        You are an expert in summarizing research papers. I'll provide you with a research paper in markdown format. Your task is to create a comprehensive summary of the paper.

        Here's what I need you to do:
        1. First Pass: Provide a brief overview of the paper (1-2 paragraphs).
        2. Second Pass: Analyze the paper's structure, methodology, and key findings (3-4 paragraphs).
        3. Third Pass: Provide a detailed analysis of the paper's contributions, limitations, and implications (4-5 paragraphs).
        4. Interpretive Summary: Synthesize the paper's core concept and significance in your own words (2-3 paragraphs).
        5. Highlights Explained: List and explain 3-5 key highlights or takeaways from the paper.
        6. Determine if this is a computer science paper, and if so, identify the specific subfield (e.g., machine learning, algorithms, systems, etc.).

        Paper:
        {markdown_content[:50000]}  # Limit to 50,000 characters to avoid token limits
        """
        
        # Generate the summary
        print(f"Generating summary using Claude...")
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=4000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        summary_content = message.content[0].text
        
        # Save the summary
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print(f"Summary saved to {summary_path}")
        
        # Extract the summary sections
        sections = {}
        current_section = None
        section_content = []
        
        for line in summary_content.split('\n'):
            if line.strip() == '':
                continue
            
            if line.startswith('# ') or line.startswith('## '):
                if current_section is not None:
                    sections[current_section] = '\n'.join(section_content)
                
                current_section = line.lstrip('#').strip()
                section_content = []
            else:
                section_content.append(line)
        
        if current_section is not None:
            sections[current_section] = '\n'.join(section_content)
        
        # Create the summary dictionary
        summary = {
            "first_pass": sections.get("First Pass", ""),
            "second_pass": sections.get("Second Pass", ""),
            "third_pass": sections.get("Third Pass", ""),
            "interpretive_summary": sections.get("Interpretive Summary", ""),
            "highlights_explained": sections.get("Highlights Explained", "")
        }
        
        # Determine if this is a computer science paper
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
        
        return summary, summary_path, is_cs_paper, cs_subfield
    
    except Exception as e:
        print(f"Error summarizing paper: {str(e)}")
        return None, None, False, "general_cs"
