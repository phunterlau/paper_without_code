import os
import argparse
from openai import OpenAI
import PyPDF2
import re
import yaml

# Initialize the OpenAI client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NIM_API_KEY")
)

def parse_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error parsing PDF: {str(e)}")
        return None

def gpt4_analyze(prompt):
    try:
        completion = client.chat.completions.create(
            model="deepseek-ai/deepseek-r1",
            temperature=0.6,
            top_p=0.7,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that analyzes scientific papers."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        
        # Accumulate the streamed response
        full_response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                print(content, end="", flush=True)
        
        print()  # New line after streaming completes
        return full_response
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        return None

def first_pass(paper_content):
    prompt = f"""
    Perform a first pass read of the following scientific paper. Focus on:
    1. Title, abstract, and introduction
    2. Section and sub-section headings
    3. Conclusions
    4. References (note any you recognize)

    After reading, answer the following questions:
    1. Category: What type of paper is this?
    2. Context: Which other papers is it related to? What theoretical bases were used?
    3. Correctness: Do the assumptions appear to be valid?
    4. Contributions: What are the paper's main contributions?
    5. Clarity: Is the paper well written?

    Paper content:
    {paper_content}

    Analysis:
    """
    return gpt4_analyze(prompt)

def second_pass(paper_content, first_pass_summary):
    prompt = f"""
    You have already performed a first pass on this paper with the following summary:

    {first_pass_summary}

    Now, perform a second pass read of the paper. Focus on:
    1. Carefully examining figures, diagrams, and other illustrations
    2. Marking unread references for further reading

    Provide a summary of the main thrust of the paper with supporting evidence.
    Highlight any parts that were difficult to understand or require background knowledge.

    Paper content:
    {paper_content}

    Analysis:
    """
    return gpt4_analyze(prompt)

def third_pass(paper_content, first_pass_summary, second_pass_summary):
    prompt = f"""
    You have already performed a first and second pass on this paper with the following summaries:

    First Pass:
    {first_pass_summary}

    Second Pass:
    {second_pass_summary}

    Now, perform a third pass read of the paper. Your goal is to virtually re-implement the paper:
    1. Identify and challenge every assumption in every statement
    2. Think about how you would present each idea
    3. Jot down ideas for future work

    Provide a detailed analysis including:
    1. The entire structure of the paper
    2. Its strong and weak points
    3. Implicit assumptions
    4. Missing citations to relevant work
    5. Potential issues with experimental or analytical techniques

    Paper content:
    {paper_content}

    Analysis:
    """
    return gpt4_analyze(prompt)

def high_level_summary(paper_content, first_pass_summary, second_pass_summary, third_pass_summary):
    prompt = f"""
    Based on the three-pass analysis of the paper, provide a high-level summary for researchers. Include:
    1. The main research question or problem addressed
    2. Key findings and contributions
    3. Methodology overview
    4. Implications for the field
    5. Potential areas for future research

    Also, highlight 3-5 key points that researchers should remember about this paper.

    First Pass Summary: {first_pass_summary}
    Second Pass Summary: {second_pass_summary}
    Third Pass Summary: {third_pass_summary}

    Paper content: {paper_content}

    High-Level Summary and Highlights:
    """
    return gpt4_analyze(prompt)

def generate_mind_map(paper_content, first_pass_summary, second_pass_summary, third_pass_summary, high_level_summary):
    prompt = f"""
    Create a vertical tree-style mind map using Mermaid syntax based on the three-pass analysis and high-level summary of the paper. Follow these instructions exactly:

    1. Start the diagram with: graph LR
    2. On the next line, add: classDef default fill:#f9f,stroke:#333,stroke-width:2px;
    3. Begin with the paper's title as the root node at the top, labeled 'root'
    4. Analyze the three-pass summaries and high-level summary to determine the main branches (key aspects) of the paper
    5. Use solid lines (-->) for main branches from the root
    6. Use dotted lines (-.->) for all sub-branches
    7. The main branches should reflect the key aspects of the paper as identified in the summaries, which may include (but are not limited to):
       - Research Question/Objective
       - Methodology
       - Key Findings/Contributions
       - Theoretical Framework
       - Data and Analysis
       - Results and Discussion
       - Implications
       - Limitations
       - Future Research Directions
    8. Use concise labels for each node, enclosed in double quotes
    9. Ensure at least 2-3 levels of hierarchy where appropriate
    10. Use this exact syntax (replace with actual content):
        root["Paper Title"]
        root --> branch1["Main Aspect 1"]
        root --> branch2["Main Aspect 2"]
        branch1 -.-> leaf1["Subtopic 1.1"]
        branch1 -.-> leaf2["Subtopic 1.2"]
        branch2 -.-> leaf3["Subtopic 2.1"]
    11. ONLY output the Mermaid syntax, nothing else

    First Pass Summary: {first_pass_summary}
    Second Pass Summary: {second_pass_summary}
    Third Pass Summary: {third_pass_summary}
    High-Level Summary: {high_level_summary}
    Paper full content: {paper_content}

    Generate the Mermaid mind map diagram:
    """
    return gpt4_analyze(prompt)

def read_yaml_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def generate_high_level_summary(paper_content, first_pass_summary, second_pass_summary, third_pass_summary, human_prompt):
    prompt = f"""
    Based on the three-pass analysis of the paper and the additional information provided, generate a high-level summary and list of highlights. The summary should:

    1. Provide an overview of the paper's main objectives, methodologies, and findings
    2. Highlight the key contributions and their significance
    3. Discuss any notable strengths or limitations
    4. Consider the additional context or information provided: {human_prompt}

    First Pass Summary: {first_pass_summary}
    Second Pass Summary: {second_pass_summary}
    Third Pass Summary: {third_pass_summary}
    Additional Information: {human_prompt}

    Generate the high-level summary and highlights:
    """
    return gpt4_analyze(prompt)

def generate_interpretive_summary(first_pass_summary, second_pass_summary, third_pass_summary, high_level_summary, human_prompt):
    prompt = f"""
    Based on the provided summaries of a research paper and the additional information, generate a short paragraph (about 150-200 words) that interprets the article for researchers and graduate students. The paragraph should:

    1. Emphasize the paper's highlights and key takeaways
    2. Explain why the paper is worth reading
    3. Discuss its potential impact or significance in the field
    4. Highlight any novel methodologies or findings
    5. Suggest potential areas for further research
    6. Consider the additional context or information provided: {human_prompt}

    Use an engaging and informative tone suitable for an academic audience. Focus on what makes this paper stand out and why it's valuable to read.

    First Pass Summary: {first_pass_summary}
    Second Pass Summary: {second_pass_summary}
    Third Pass Summary: {third_pass_summary}
    High-Level Summary: {high_level_summary}
    Additional Information: {human_prompt}

    Generate the interpretive summary paragraph:
    """
    return gpt4_analyze(prompt)

def generate_highlights_explained(paper_content, first_pass_summary, second_pass_summary, third_pass_summary, human_prompt):
    prompt = f"""
    Based on the three-pass analysis of the paper and the additional information provided, generate a "Highlights Explained" section. This section should:

    1. Identify 4-5 key technical highlights, advantages, uniqueness or central concepts from the paper
    2. For each highlight or concept:
       a. Provide a clear, concise explanation of what it means
       b. Explain why it's significant or important in the context of the paper
       c. If applicable, briefly mention how it relates to existing work or its potential impact

    3. Consider the additional context or information provided in framing these explanations: {human_prompt}

    Format the output as a list, with each highlight as a subheading followed by its explanation.

    First Pass Summary: {first_pass_summary}
    Second Pass Summary: {second_pass_summary}
    Third Pass Summary: {third_pass_summary}
    Additional Information: {human_prompt}
    Paper full content: {paper_content}

    Generate the Highlights Explained section in Markdown format, and the top level is heading 2 (##) and subheadings are heading 3 (###) and going down:
    """
    return gpt4_analyze(prompt)

def generate_markmap_mindmap(paper_content, first_pass_summary, second_pass_summary, third_pass_summary, high_level_summary):
    prompt = f"""
    Create a hierarchical mind map using Markdown syntax based on the three-pass analysis and high-level summary of the paper. Follow these instructions exactly:

    1. Start with the paper's title as the root node at the top level (#)
    2. Use the following structure as the main branches if available, add other structures if needed(##):
       - Research Question/Objective
       - Methodology
       - Key Findings/Contributions
       - Theoretical Framework
       - Results and Discussion
       - Implications
       - Limitations
       - Future Research Directions
    3. Under each main branch, add relevant sub-topics as third-level headings (###) and fourth-level headings (####) if necessary
    4. Use bullet points to add brief explanations or key points under each heading
    5. Ensure the structure is detailed and reflects the content of the paper
    6. Use concise labels for each node, but if the text is long, it's okay - it will be wrapped automatically
    7. Do not use any code blocks or backticks in the output

    First Pass Summary: {first_pass_summary}
    Second Pass Summary: {second_pass_summary}
    Third Pass Summary: {third_pass_summary}
    High-Level Summary: {high_level_summary}
    Paper full content: {paper_content}

    Generate the Markmap-compatible Markdown structure:
    """
    markmap_content = gpt4_analyze(prompt)
    
    # Add YAML front matter
    yaml_front_matter = """---
title: Paper Mindmap
markmap:
  colorFreezeLevel: 5
  initialExpandLevel: -1
  maxWidth: 300
---

"""
    return yaml_front_matter + markmap_content

def save_markmap_to_file(file_path, markmap_content):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = f"{base_name}-markmap.md"
    
    with open(output_file, 'w') as f:
        f.write(markmap_content)
    
    print(f"\nMarkmap saved to {output_file}")

def summarize_paper(file_path, paper_content, human_prompt):
    print("Performing first pass...")
    first_pass_summary = first_pass(paper_content)
    print("\nFirst pass summary:")
    print(first_pass_summary)

    print("\nPerforming second pass...")
    second_pass_summary = second_pass(paper_content, first_pass_summary)
    print("\nSecond pass summary:")
    print(second_pass_summary)

    print("\nPerforming third pass...")
    third_pass_summary = third_pass(paper_content, first_pass_summary, second_pass_summary)
    print("\nThird pass summary:")
    print(third_pass_summary)

    print("\nGenerating Highlights Explained section...")
    highlights_explained = generate_highlights_explained(paper_content, first_pass_summary, second_pass_summary, third_pass_summary, human_prompt)
    print("\nHighlights Explained:")
    print(highlights_explained)

    print("\nGenerating interpretive summary...")
    interpretive_summary = generate_interpretive_summary(first_pass_summary, second_pass_summary, third_pass_summary, highlights_explained, human_prompt)
    print("\nInterpretive summary:")
    print(interpretive_summary)

    print("\nGenerating Mermaid mind map...")
    mermaid_mind_map = generate_mind_map(paper_content, first_pass_summary, second_pass_summary, third_pass_summary, highlights_explained)
    print("\nMermaid mind map generated")

    print("\nGenerating Markmap mind map...")
    markmap_mind_map = generate_markmap_mindmap(paper_content, first_pass_summary, second_pass_summary, third_pass_summary, highlights_explained)
    print("\nMarkmap mind map generated")

    # Save Markmap to a separate file
    save_markmap_to_file(file_path, markmap_mind_map)

    return {
        "first_pass": first_pass_summary,
        "second_pass": second_pass_summary,
        "third_pass": third_pass_summary,
        "highlights_explained": highlights_explained,
        "interpretive_summary": interpretive_summary,
        "mermaid_mind_map": mermaid_mind_map,
        "markmap_mind_map": markmap_mind_map
    }

def save_summary_to_markdown(file_path, summary):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = f"{base_name}-summary.md"
    
    with open(output_file, 'w') as f:
        f.write(f"# Summary of {base_name}\n\n")
        f.write("## Interpretive Summary for Researchers and Graduate Students\n\n")
        f.write(f"{summary['interpretive_summary']}\n\n")
        f.write("## Highlights Explained\n\n")
        f.write(f"{summary['highlights_explained']}\n\n")
        f.write("## Mermaid Mind Map\n\n")
        f.write("```mermaid\n")
        f.write(f"{summary['mermaid_mind_map']}\n")
        f.write("```\n\n")
        f.write("## First Pass\n\n")
        f.write(f"{summary['first_pass']}\n\n")
        f.write("## Second Pass\n\n")
        f.write(f"{summary['second_pass']}\n\n")
        f.write("## Third Pass\n\n")
        f.write(f"{summary['third_pass']}\n")
    
    print(f"\nSummary saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Summarize a scientific paper using the three-pass approach.")
    parser.add_argument("yaml_file", help="Path to the YAML configuration file")
    args = parser.parse_args()

    config = read_yaml_config(args.yaml_file)
    file_path = config['file_name']
    human_prompt = config.get('human_prompt', '')

    try:
        paper_content = parse_pdf(file_path)
        if paper_content:
            summary = summarize_paper(file_path, paper_content, human_prompt)
            save_summary_to_markdown(file_path, summary)
            print("\nComplete summary generated and saved successfully.")
        else:
            print("Failed to extract text from the PDF.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
