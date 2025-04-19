import random
from dotenv import load_dotenv
import os
from typing import Dict
import openai

from prompt_example_selector import initialize_question_clusters, \
    prompt_example_generator
from utils import format_table, construct_chain_of_thought
from preprocessor import preprocess_dataset
from data_loader import download_data
import shared_data

# Global variable to hold the max_samples value
MAX_SAMPLES = 3037


# =============================================================================
# Step 1: Preprocess Examples
# =============================================================================
def load_and_preprocess_data(url: str, max_samples: int = None) -> None:
    """
    Load and preprocess the data only once at the beginning.
    Args:
        url: Data source URL
        max_samples: Maximum samples to process (defaults to global
        MAX_SAMPLES)
    """
    # Use global MAX_SAMPLES if no override provided
    if max_samples is None:
        max_samples = MAX_SAMPLES

    # Check if data is already loaded and processed
    if not shared_data.processed_dataset:
        # Load and preprocess data only once
        print("Loading and preprocessing data...")
        data = download_data(url)
        shared_data.processed_dataset = preprocess_dataset(data, MAX_SAMPLES)
        print(
            f"✅ After preprocessing: {len(shared_data.processed_dataset)} "
            f"examples loaded.")
        if len(shared_data.processed_dataset) == 0:
            print("❌ preprocess_dataset() returned an empty dictionary!")
        print("Data preprocessing complete.")
    else:
        print("Data already loaded and preprocessed. Skipping...")

    if (not shared_data.question_to_cluster_label or not
    shared_data.cluster_idx_to_questions):
        print("Initializing clustering...")
        initialize_question_clusters()
    else:
        print("Clustering already initialized.")

# =============================================================================
# Step 2: Set up LangChain Prompt Template
# =============================================================================
# Initialize OpenAI API Key
load_dotenv()  # This loads variables from .env into the environment
openai.api_key = os.getenv("OPENAI_API_KEY")


# Define a function for querying the processed data
def query_data(question: str, processed_dataset: Dict) -> str:
    """
    Given a question, return the associated context from the preprocessed data.
    """
    return processed_dataset.get(question,
                              {"error": "Data for this question not found."})


# Step 3: Create a LangChain Prompt Template
# Define a prompt template to generate an answer or code based on the
# question and context.
def generate_few_shot_prompt(processed_data, user_question, context, n=3, ):
    # all_questions = list(processed_data.keys())
    # random.seed(42)  # Set a fixed seed for reproducibility
    # selected_questions = random.sample(all_questions, n)
    top_num = 3
    selected_questions = prompt_example_generator(user_question, top_num)

    examples = []
    for idx, example_question in enumerate(selected_questions):
        data = processed_data[example_question]
        table = format_table(data["table"])
        reasoning_steps = construct_chain_of_thought(data)
        output = (f"Program: {data.get('program')}\nAnswer: "
                  f"{data.get('answer')}")

        example_prompt = f"""
Example: {idx + 1}
Question: {example_question}
Pre-context: 
{data.get("pre_text", "")}

Table: 
{table}

Post-context: 
{data.get("post_text", "")}

Let's think step by step:
{reasoning_steps}

Output: 
{output}
"""
        examples.append(example_prompt.strip())

    if "error" in context:
        return context["error"]
    user_question_pre_text = context["pre_text"]
    user_question_table = format_table(context["table"])
    user_question_post_text = context["post_text"]
    question_prompt = f""" 
Questions

You are a helpful financial analysis assistant. Using the pre_text, table
and post_text and reasoning details below and examples provided above, 
please write a program that calculates the answer to the question and then 
provides the final answer.
       
1. Analyze this question: {user_question}
2. Use this table and text data:
Pre_text: 
{user_question_pre_text}
Table: 
{user_question_table}
Post_text:
{user_question_post_text} 
3. Please use the examples above "Let's think step by step" to do reason and 
calculation

4. Please produce the following outputs:
- Reasoning Steps:
- Program: function-style operations or function call expressions
- Answer: Just the final value as string with max.2 digits decimal
- Confidence: 0-100% certainty 

**Example Output Format:**
Program: such as "multiply(2.12, const_1000), add(#0, 112), greater(#0, 5), 
Prefer subtract(x,const_100) over subtract(x,100)"
Answer: such as "5.2", "-4.9%", "8.92%", "$ 378.7 million", "2232", "no", "yes"
Confidence: 92%

       """

    final_prompt = "\n\n---\n\n".join(
        examples) + "\n\n---\n\n" + question_prompt
    return final_prompt


# =============================================================================
# Step 3: Initialize the LLM
# =============================================================================
def query_gpt(prompt: str) -> str:
    """
        Sends a prompt to the GPT-3.5-turbo model and returns the generated
        response.
        """
    client = openai.OpenAI()  # Create a client instance

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "You are a helpful financial analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()


# =============================================================================
# Step 4: Generate Output
# ============================================================================
def generate_answer(question: str) -> str:
    """
    Generate the answer for a given question using preprocessed data.
    """
    # Ensure the data is loaded and preprocessed before inference
    if not shared_data.processed_dataset:
        raise ValueError("Data must be loaded and preprocessed first.")

    # Retrieve context for the given question
    context = query_data(question, shared_data.processed_dataset)
    few_shot_prompt = generate_few_shot_prompt(shared_data.processed_dataset,
                                               question,
                                               context, n=4)
    print("------ Prompt Sent to GPT ------\n", few_shot_prompt)
    response = query_gpt(few_shot_prompt)
    print("------ GPT-3.5 Response ------\n", response)
    return response


def generate_ground_truth(question: str) -> Dict[str, str]:
    """
    Generate the ground truth (program and answer) for a given question.
    """
    # Ensure the data is loaded and preprocessed before inference
    if not shared_data.processed_dataset:
        raise ValueError("Data must be loaded and preprocessed first.")

    # Retrieve context for the given question
    context = query_data(question, shared_data.processed_dataset)
    # Retrieve the expected program and answer from the context
    program = context.get("program", "Not Available")
    answer = context.get("answer", "Not Available")

    # Check if either value is still None or empty, and provide a fallback
    # if necessary
    if program is None or program == "":
        program = "Program not found"
    if answer is None or answer == "":
        answer = "Answer not found"

    ground_truth = {
        "Program": program,
        "Answer": answer
    }

    return ground_truth


if __name__ == "__main__":
    # URL to download the data
    url = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"

    # First time: Load and preprocess the data
    load_and_preprocess_data(url)

    # Running inference for a single question
    question_text = "what was the percent of the growth in the revenues from 2007 to 2008"
    print("\n------ GPT-3.5 Response ------\n")
    generate_answer(question_text)

    print("\n------ Ground Truth ------")
    ground_truth = generate_ground_truth(question_text)
    print("Expected Program:", ground_truth["Program"])
    print("Expected Answer:", ground_truth["Answer"])
