"""
Module for downloading PDFs from arXiv.
"""
import os
import re
import requests
from urllib.parse import urlparse

def extract_arxiv_id(url):
    """
    Extract the arXiv ID from a URL.
    
    Args:
        url (str): The arXiv URL
        
    Returns:
        str: The arXiv ID
    """
    # Extract the arXiv ID from the URL
    if "arxiv.org/abs/" in url:
        # Extract the ID from the URL
        arxiv_id = url.split("arxiv.org/abs/")[1].split("/")[0].split("v")[0]
        return arxiv_id
    elif "arxiv.org/pdf/" in url:
        # Extract the ID from the PDF URL
        arxiv_id = url.split("arxiv.org/pdf/")[1].split(".pdf")[0].split("v")[0]
        return arxiv_id
    else:
        # Try to extract the ID using a regular expression
        match = re.search(r"(\d+\.\d+)", url)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Could not extract arXiv ID from URL: {url}")

def download_pdf(url, output_dir="output"):
    """
    Download a PDF from an arXiv URL.
    
    Args:
        url (str): The arXiv URL
        output_dir (str): The directory to save the PDF to
        
    Returns:
        tuple: (pdf_path, arxiv_id, workflow_dir)
    """
    try:
        # Extract the arXiv ID from the URL
        arxiv_id = extract_arxiv_id(url)
        
        # Clean the ID for use in filenames
        clean_id = arxiv_id.replace('/', '_')
        
        # Create the workflow directory
        workflow_dir = os.path.join(output_dir, "workflows", clean_id)
        os.makedirs(workflow_dir, exist_ok=True)
        
        # Define the PDF path
        pdf_path = os.path.join(workflow_dir, f"{clean_id}.pdf")
        
        # Check if the PDF already exists
        if os.path.exists(pdf_path):
            print(f"PDF already exists at {pdf_path}")
            return pdf_path, arxiv_id, workflow_dir
        
        # Construct the PDF URL
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        # Download the PDF
        print(f"Downloading PDF from {pdf_url}...")
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        
        # Save the PDF
        with open(pdf_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"PDF downloaded to {pdf_path}")
        return pdf_path, arxiv_id, workflow_dir
    
    except Exception as e:
        print(f"Error downloading PDF: {str(e)}")
        return None, None, None
