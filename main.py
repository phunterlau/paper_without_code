#!/usr/bin/env python3
"""
PaperWoCode: A tool to generate code prototypes from research papers.
"""
import argparse
import os
import sys
from paperwocode.agent import run_agent_workflow

def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Generate code from arXiv papers")
    parser.add_argument("arxiv_url", nargs="?", default="https://arxiv.org/abs/2411.16905", 
                        help="URL to the arXiv paper (default: https://arxiv.org/abs/2411.16905)")
    parser.add_argument("--force", action="store_true", help="Force code generation even for non-CS papers")
    parser.add_argument("--output", "-o", default="output", help="Output directory for generated files")
    parser.add_argument("--iterations", "-i", type=int, default=3, help="Maximum number of iterations for code refinement")
    parser.add_argument("--no-cache", action="store_true", help="Skip using cached results and rerun all steps")
    args = parser.parse_args()
    
    # Check for API keys
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Run the agent workflow
    run_agent_workflow(args.arxiv_url, args.force, args.output, args.iterations, args.no_cache)

if __name__ == "__main__":
    main()
