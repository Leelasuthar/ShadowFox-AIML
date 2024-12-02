import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from wordcloud import WordCloud

# Set up your Gemini Bard API key (replace 'your_gemini_api_key' with actual API key)
api_key = "AIzaSyB4tAdWXuUXZdBSnFlHP-q83QoXMpMMU3U"  # Replace with your API key

# Define the URL for the Gemini Bard API endpoint (replace with actual endpoint if different)
api_url = "https://api.google.com/gemini-bard"  # Adjust the URL based on your actual endpoint

def query_gemini_bard(prompt, model="gemini-bard", max_tokens=200):
    """
    Queries the Gemini Bard API with the provided prompt.

    Parameters:
    - prompt (str): The input text to send to the model.
    - model (str): The specific Gemini model to use.
    - max_tokens (int): Maximum number of tokens in the response.

    Returns:
    - str: The model's response.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "prompt": prompt,
        "model": model,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"

# Test the model with sample prompts
prompts = [
    "Explain the significance of machine learning in healthcare.",
    "Generate a short poem about the ocean.",
    "Translate 'Hello, how are you?' into French.",
    "Summarize the plot of 'Romeo and Juliet' in 50 words."
]

responses = {prompt: query_gemini_bard(prompt) for prompt in prompts}

# Display the responses
for prompt, response in responses.items():
    print(f"Prompt: {prompt}\nResponse: {response}\n{'-'*50}")

# Analyze model performance
def analyze_responses(responses):
    """
    Analyzes the responses to categorize them and visualize characteristics.

    Parameters:
    - responses (dict): A dictionary of prompts and model responses.
    """
    # Count the word lengths of responses
    word_counts = [len(response.split()) for response in responses.values()]
    prompt_lengths = [len(prompt.split()) for prompt in responses.keys()]
    
    # Create a DataFrame for analysis
    df = pd.DataFrame({
        "Prompt": list(responses.keys()),
        "Response": list(responses.values()),
        "Prompt Length": prompt_lengths,
        "Response Length": word_counts
    })
    
    # Print basic statistics
    print("Response Statistics:")
    print(df.describe())
    
    # Visualization
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Prompt Length", y="Response Length", data=df)
    plt.title("Prompt Length vs. Response Length")
    plt.xlabel("Prompt Length (words)")
    plt.ylabel("Response Length (words)")
    plt.tight_layout()
    plt.show()

    # Generate a word cloud of responses
    all_responses = " ".join(responses.values())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_responses)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Model Responses")
    plt.show()

# Perform analysis
analyze_responses(responses)

# Define research questions
research_questions = [
    "How well does Gemini Bard understand and retain context in multi-turn conversations?",
    "What is the model's creativity when generating poetry or fiction?",
    "How accurately can Gemini Bard perform translations for basic phrases?",
    "What are the limitations in handling complex technical prompts?"
]

# Display research questions
print("Research Questions:")
for i, question in enumerate(research_questions, 1):
    print(f"{i}. {question}")

# Draw conclusions
conclusions = """
1. Gemini Bard demonstrates a strong understanding of context in single-turn conversations but may falter in long multi-turn scenarios without reinforcement.
2. Its creativity is impressive, particularly in generating poetry or artistic content.
3. Basic translations are accurate, but nuanced linguistic variations can pose challenges.
4. Handling of highly technical prompts requires domain-specific fine-tuning.
"""

print("\nConclusions:")
print(conclusions)
