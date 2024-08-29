import os
import openai
from openai import OpenAI
import json

# Set up OpenAI client
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Simulated physical schema
# This represents the actual database structure, similar to what's described in the blog post
physical_schema = {
    "tables": [
        {"name": "sales_facts", "columns": ["date_id", "product_id", "revenue", "quantity"]},
        {"name": "product_dim", "columns": ["product_id", "product_name", "category_id"]},
        {"name": "date_dim", "columns": ["date_id", "date", "month", "year"]},
        {"name": "category_dim", "columns": ["category_id", "category_name"]}
    ]
}

# Simulated semantic model
# This represents the business-friendly view of the data, as mentioned in the blog post
# It bridges the gap between business terms and the actual database schema
semantic_model = {
    "tables": [
        {"name": "sales", "columns": ["date", "product", "revenue", "quantity"]},
        {"name": "products", "columns": ["product_id", "name", "category"]}
    ],
    "metrics": [
        {"name": "total_revenue", "definition": "SUM(sales.revenue)"},
        {"name": "total_quantity", "definition": "SUM(sales.quantity)"}
    ]
}

# Simulated verified queries
# These represent known-good queries that can be used as reference, similar to the blog post's mention of verified queries
verified_queries = [
    "SELECT product, SUM(revenue) as total_revenue FROM sales GROUP BY product ORDER BY total_revenue DESC LIMIT 1",
    "SELECT DATE_TRUNC('month', date) as month, SUM(revenue) as monthly_revenue FROM sales GROUP BY month ORDER BY month"
]

def classification_agent(question):
    """
    This function represents the Classification Agent mentioned in the blog post.
    It categorizes the input question to ensure only appropriate queries are processed.
    
    Future improvements:
    - Implement more fine-grained categories for different types of SQL questions
    - Use a more sophisticated model or a custom-trained classifier for better accuracy
    """
    prompt = f"""Classify the following question into one of these categories:
    1. SQL_QUESTION: An unambiguous question that can be answered with SQL
    2. AMBIGUOUS: A question that is too vague or ambiguous to be answered with SQL
    3. NON_DATA: A question that is not related to data or cannot be answered with a database query
    4. NON_SQL: A data-related question that cannot be directly answered with SQL (e.g., requests for visualizations)

    Examples:
    - "What was the total revenue last month?" -> SQL_QUESTION
    - "How has our monthly revenue changed over the past year?" -> SQL_QUESTION
    - "Show me the sales data" -> AMBIGUOUS
    - "What's the weather like today?" -> NON_DATA
    - "Can you create a visualization of our sales data?" -> NON_SQL

    Question: {question}

    Classification:"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def feature_extraction_agent(question):
    """
    This function represents the Feature Extraction Agent mentioned in the blog post.
    It identifies important aspects of the question to guide SQL generation.
    
    Future improvements:
    - Expand the set of features to cover more query characteristics
    - Implement a more robust parsing mechanism for complex questions
    """
    prompt = f"""Extract relevant features from the following question. For each feature, respond with YES or NO:

    1. Is it a time-series question?
    2. Does it involve period-over-period calculation?
    3. Does it involve ranking?
    4. What metrics are being asked about? (List them)

    Question: {question}

    Please format your response as follows:
    Time-series: YES/NO
    Period-over-period: YES/NO
    Ranking: YES/NO
    Metrics: [list of metrics]
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # Parse the response
    content = response.choices[0].message.content.strip()
    features = {}
    for line in content.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()
            if value.upper() == 'YES':
                features[key] = True
            elif value.upper() == 'NO':
                features[key] = False
            elif key == 'metrics':
                features[key] = [m.strip() for m in value.strip('[]').split(',')]
            else:
                features[key] = value

    return features

def generate_logical_schema(physical_schema, semantic_model):
    """
    This function simulates the creation of a logical schema, which is a key concept in the blog post.
    It simplifies the physical schema based on the semantic model, making it easier for the SQL generation step.
    
    Future improvements:
    - Implement a more sophisticated mapping between physical and logical schemas
    - Consider more aspects of the semantic model in the schema generation
    """
    # In a real system, this would be more complex and would consider the semantic model
    # For this prototype, we'll use a simplified version
    logical_schema = {
        "tables": [
            {
                "name": "sales",
                "columns": ["date", "product", "revenue", "quantity"],
                "mapping": "SELECT date_dim.date, product_dim.product_name, sales_facts.revenue, sales_facts.quantity FROM sales_facts JOIN date_dim ON sales_facts.date_id = date_dim.date_id JOIN product_dim ON sales_facts.product_id = product_dim.product_id"
            },
            {
                "name": "products",
                "columns": ["product_id", "name", "category"],
                "mapping": "SELECT product_id, product_name, category_dim.category_name FROM product_dim JOIN category_dim ON product_dim.category_id = category_dim.category_id"
            }
        ]
    }
    return logical_schema

def sql_generation_agent(question, logical_schema, features):
    """
    This function represents the SQL Generation Agent mentioned in the blog post.
    It uses the logical schema and extracted features to create an initial SQL query.
    
    Future improvements:
    - Implement the two-step SQL generation process described in the blog post
    - Use multiple specialized models for different types of SQL queries
    """
    prompt = f"""Given the following logical schema and question features, generate a SQL query to answer the question.

    Logical Schema:
    {json.dumps(logical_schema, indent=2)}

    Question Features:
    {json.dumps(features, indent=2)}

    Question: {question}

    SQL Query (using logical schema):"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def post_process_sql(logical_sql, logical_schema, physical_schema):
    """
    This function represents the post-processing step mentioned in the blog post.
    It converts the SQL query from the logical schema to the physical schema.
    
    Future improvements:
    - Implement more sophisticated query transformation techniques
    - Handle complex join conditions and subqueries more effectively
    """
    prompt = f"""Given the following logical SQL query and the mappings between logical and physical schemas, 
    rewrite the SQL query to work with the physical schema.

    Logical SQL:
    {logical_sql}

    Logical to Physical Mapping:
    {json.dumps(logical_schema, indent=2)}

    Physical Schema:
    {json.dumps(physical_schema, indent=2)}

    Rewritten SQL for Physical Schema:"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def error_correction_agent(sql_query):
    """
    This function represents the Error Correction Agent mentioned in the blog post.
    It checks for and corrects errors in the generated SQL query.
    
    Future improvements:
    - Implement actual SQL parsing and validation
    - Use a database-specific validator to catch syntax errors
    """
    # Simulated error checking (in a real system, this would involve actual SQL parsing and validation)
    prompt = f"""Check the following SQL query for errors and correct them if necessary:

    SQL Query:
    {sql_query}

    Corrected SQL Query:"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def synthesizer_agent(question, sql_query, verified_queries):
    """
    This function represents the Synthesizer Agent mentioned in the blog post.
    It combines the generated SQL with verified queries to produce a final, accurate query.
    
    Future improvements:
    - Implement more sophisticated query merging techniques
    - Consider query performance and optimization in the synthesis process
    """
    prompt = f"""Given the original question, generated SQL query, and a set of verified queries, synthesize a final SQL query that accurately answers the question.

    Question: {question}

    Generated SQL Query:
    {sql_query}

    Verified Queries:
    {json.dumps(verified_queries, indent=2)}

    Final SQL Query:"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def answer_question(question):
    """
    This function orchestrates the entire process of answering a question,
    following the workflow described in the blog post.
    
    Future improvements:
    - Implement more robust error handling and fallback mechanisms
    - Add support for iterative refinement based on user feedback
    - Incorporate additional context enrichment steps as mentioned in the blog post
    """
    print(f"Question: {question}")

    # Step 1: Classification
    classification = classification_agent(question)
    print(f"Classification: {classification}")

    if classification != "SQL_QUESTION":
        return f"I'm sorry, but I can only answer unambiguous SQL questions. This question was classified as: {classification}"

    # Step 2: Feature Extraction
    try:
        features = feature_extraction_agent(question)
        print(f"Extracted Features: {features}")
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        return "I encountered an error while analyzing the question. Please try rephrasing it."

    # Step 3: Generate Logical Schema
    logical_schema = generate_logical_schema(physical_schema, semantic_model)
    print(f"Logical Schema: {json.dumps(logical_schema, indent=2)}")

    # Step 4: SQL Generation (using logical schema)
    logical_sql = sql_generation_agent(question, logical_schema, features)
    print(f"Generated Logical SQL: {logical_sql}")

    # Step 5: Post-process SQL (convert to physical schema)
    physical_sql = post_process_sql(logical_sql, logical_schema, physical_schema)
    print(f"Post-processed Physical SQL: {physical_sql}")

    # Step 6: Error Correction
    corrected_sql = error_correction_agent(physical_sql)
    print(f"Corrected SQL: {corrected_sql}")

    # Step 7: Synthesis
    final_sql = synthesizer_agent(question, corrected_sql, verified_queries)
    print(f"Final SQL: {final_sql}")

    return final_sql

if __name__ == "__main__":
    # Test questions covering various scenarios
    questions = [
        "What was the best-selling product last month?",
        "How has our monthly revenue changed over the past year?",
        "What's the total revenue for each product category in Q1 2024?",
        "Who are our top 5 customers by total spending?",
        "What's the year-over-year growth rate for each product?",
        "Which products have consistently increased in sales every month this year?",
        "What's the average order value for each customer segment?",
        # this one is a bit ambiguous and not yet answered well, need to improve the model by training on more data
        "How does the revenue distribution vary by day of the week?",
        "What percentage of our total revenue comes from new customers vs. returning customers?",
        "Which products are often bought together?",
        "What's our customer retention rate by quarter?",
        "How does the profit margin vary across different product categories and regions?",
        "What's the trend in average transaction value over the last 6 months?",
        "Which sales representatives have exceeded their quarterly targets?",
        "What's the weather like today?",  # Non-data question
        "Can you create a visualization of our sales data?",  # Non-SQL data question
        "Show me the sales data",  # Ambiguous question
    ]

    for question in questions:
        print("\n" + "="*50)
        result = answer_question(question)
        print(f"Result: {result}")
        print("="*50 + "\n")

# Overall future improvements:
# 1. Implement the full two-step SQL generation process described in the blog post
# 2. Use multiple specialized models for different types of SQL queries and different stages of the process
# 3. Incorporate more sophisticated error handling and query optimization techniques
# 4. Implement a feedback loop to continuously improve the system based on user interactions
# 5. Add support for more complex analytics, including time series analysis and predictive modeling
# 6. Enhance the semantic model to capture more business logic and domain-specific knowledge
# 7. Implement a more robust method for handling ambiguous questions and providing suggestions for clarification