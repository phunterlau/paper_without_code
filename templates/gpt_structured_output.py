"""
Template for LLM-related papers using GPT for structured output.
"""
import os
import json
import argparse
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_structured_output(prompt, output_schema, model="gpt-4o-mini"):
    """
    Generate structured output using GPT.
    
    Args:
        prompt (str): The prompt to send to GPT
        output_schema (dict): The schema for the output
        model (str): The GPT model to use
        
    Returns:
        dict: The structured output
    """
    # Create the system message with the output schema
    system_message = f"""
    You are a helpful AI assistant that generates structured output according to a specific schema.
    
    The output should be valid JSON that follows this schema:
    {json.dumps(output_schema, indent=2)}
    
    Respond ONLY with the JSON output, without any additional text or explanations.
    """
    
    # Generate the response
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )
    
    # Parse the response
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        print("Error: Failed to parse JSON response")
        return None

def main():
    """
    Main function to demonstrate the implementation.
    """
    parser = argparse.ArgumentParser(description="Generate structured output using GPT")
    parser.add_argument("--prompt", type=str, default="Summarize the key points of the latest research in large language models", help="The prompt to send to GPT")
    args = parser.parse_args()
    
    # Define the output schema
    output_schema = {
        "type": "object",
        "properties": {
            "key_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"}
                    }
                }
            },
            "summary": {"type": "string"},
            "relevance": {
                "type": "object",
                "properties": {
                    "academic": {"type": "number", "minimum": 0, "maximum": 10},
                    "industry": {"type": "number", "minimum": 0, "maximum": 10}
                }
            }
        }
    }
    
    # Generate the structured output
    result = generate_structured_output(args.prompt, output_schema)
    
    # Print the result
    if result:
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
