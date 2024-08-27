import os
from openai import OpenAI
import PyPDF2

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that analyzes scientific papers."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in GPT-4 analysis: {str(e)}")
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

def second_pass(paper_content):
    prompt = f"""
    Perform a second pass read of the following scientific paper. Focus on:
    1. Carefully examining figures, diagrams, and other illustrations
    2. Marking unread references for further reading

    Provide a summary of the main thrust of the paper with supporting evidence.
    Highlight any parts that were difficult to understand or require background knowledge.

    Paper content:
    {paper_content}

    Analysis:
    """
    return gpt4_analyze(prompt)

def third_pass(paper_content):
    prompt = f"""
    Perform a third pass read of the following scientific paper. Your goal is to virtually re-implement the paper:
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

def summarize_paper(paper_content):
    print("Performing first pass...")
    first_pass_summary = first_pass(paper_content)
    print("\nFirst pass summary:")
    print(first_pass_summary)

    print("\nPerforming second pass...")
    second_pass_summary = second_pass(paper_content)
    print("\nSecond pass summary:")
    print(second_pass_summary)

    print("\nPerforming third pass...")
    third_pass_summary = third_pass(paper_content)
    print("\nThird pass summary:")
    print(third_pass_summary)

    return {
        "first_pass": first_pass_summary,
        "second_pass": second_pass_summary,
        "third_pass": third_pass_summary
    }

if __name__ == "__main__":
    file_path = input("Please enter the path to the PDF file: ")
    try:
        paper_content = parse_pdf(file_path)
        if paper_content:
            summary = summarize_paper(paper_content)
            print("\nComplete summary generated successfully.")
        else:
            print("Failed to extract text from the PDF.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")