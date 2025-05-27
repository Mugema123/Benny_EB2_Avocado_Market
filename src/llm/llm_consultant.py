# src/llm/llm_consultant.py

import ollama

def consult_llm_with_metrics(metrics: dict, feature_columns: list, model_name='llama2'):
    # Format the metrics into a readable string
    metric_summary = "\n".join([f"{key}: {value:.2f}" for key, value in metrics.items()])

    # Convert list of features to string
    feature_list = ", ".join(feature_columns)

    # Build the dynamic prompt
    prompt = f"""
You are an AI  consultant assisting with a Avocado Market in the USA: Price & Sales Analysis.
The machine learning model uses the following input features:
{feature_list}

Here are the current model performance metrics:

{metric_summary}

Use Case: Cash Flow Forecasting, Customer Retention, Performance Benchmarking:

1. Cash Flow Forecasting & Financial Health:
            Please answer the following:
        What specific financial patterns in operations are putting pressure on short-term cash flow?
        Which immediate actions should the CFO or operations manager take in the next 30–60 days to stabilize cash flow?
        How can we use its current cash runway forecast to inform operational decisions during the peak harvest season?
        Your response must be structured as:
        Summary of key cash flow risks (specific to agri-logistics)
        Top 3 urgent financial decisions for the team
        Actionable tips for interpreting their cash dashboard (designed in Power BI)


2. Customer Retention & Churn Prevention:
            Please provide:
        Based on our customer history, which behaviors or transactional signals most strongly indicate customer churn?
        What low-cost customer success strategies could we use to retain small farm clients who are at risk of switching to competitors?
        How should the sales and support teams prioritize outreach using the ML-generated churn risk scores?
        Your answer should include:
        List of top churn flags (based on internal CRM and support data)


3. Operational Benchmarking & Performance Monitoring:
   
            Your task:
        How can we use benchmarking insights to make competitive decisions?
        Which three KPIs should the operations manager track weekly in the custom dashboard to ensure real-time health monitoring?
        What patterns or thresholds in these KPIs should alert the team to operational inefficiencies or rising customer dissatisfaction?
        Output format:
        Benchmarking strategy summary (3 bullet points)
        Top 3 KPIs to track every week (with ideal target ranges)
        Red flags or trends to investigate immediately

Ensure your suggestions are specific, practical, and understandable to  stakeholders.
"""

    # Send prompt to the LLM
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']
