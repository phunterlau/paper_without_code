"""
Module for converting PDFs to markdown.
"""
import os
from markitdown import MarkItDown

def convert_pdf_to_markdown(pdf_path, output_dir):
    """
    Convert a PDF to markdown.
    
    Args:
        pdf_path (str): The path to the PDF file
        output_dir (str): The directory to save the markdown to
        
    Returns:
        tuple: (markdown_path, markdown_content)
    """
    try:
        # Define the markdown path
        markdown_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}.md")
        
        # Check if the markdown already exists
        if os.path.exists(markdown_path):
            print(f"Markdown already exists at {markdown_path}")
            with open(markdown_path, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            return markdown_path, markdown_content
        
        # Convert the PDF to markdown
        print(f"Converting PDF to markdown...")
        md = MarkItDown(enable_plugins=False)
        result = md.convert(pdf_path)
        
        # Save the markdown
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(result.text_content)
        
        print(f"Markdown saved to {markdown_path}")
        return markdown_path, result.text_content
    
    except Exception as e:
        print(f"Error converting PDF to markdown: {str(e)}")
        return None, None
