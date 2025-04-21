from dotenv import load_dotenv
import os
from typing import Dict
import openai
from src.prompt_LLM.prompt_shots_selector import prompt_example_generator
from src.utils.utils import format_table, construct_chain_of_thought
from src.shared import shared_data

# =============================================================================
# Step 1: Set up LangChain Prompt Template
# =============================================================================
# Initialize OpenAI API Key
load_dotenv()  # This loads variables from .env into the environment
openai.api_key = os.getenv("OPENAI_API_KEY")


# Define a function for querying the processed data
def query_data(question: str, processed_dataset: Dict) -> dict:
    """
    Given a question, return the associated context from the preprocessed data.
    """
    return processed_dataset.get(question, {"error": "Data for this question "
                                                     "not found."})


# Step 3: Create a LangChain Prompt Template

def generate_few_shot_prompt(processed_data, user_question, context, num_example):
    """
    Generate a few-shot prompt by constructing example prompts.

    Parameters:
        processed_data (Dict): A dictionary containing preprocessed data for
        each question.
        user_question (str): The user's question that needs to be answered.
        context (Dict): A dictionary containing context for the user
        question, including pre_text, table, post_text, and any potential error
        num_example: number of shots

    Returns:
        str: The final prompt string formatted for few-shot learning.
    """
    # Check if the processed data is empty
    if "error" in context:
        return context["error"]

    # Generate the top N most relevant questions for the given user question
    selected_questions = prompt_example_generator(user_question, num_example)

    # Initialize an empty list to hold the formatted example prompts
    examples = []

    # Loop through the selected questions and format them into examples
    for idx, example_question in enumerate(selected_questions):
        data = processed_data[example_question]
        table = format_table(data["table"])
        reasoning_steps = construct_chain_of_thought(data)
        output = (f"Program: {data.get('program')}\nAnswer: "
                  f"{data.get('answer')}")

        # Create a formatted example prompt string
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
        # Append the formatted example to the example list
        examples.append(example_prompt.strip())

    # If an error exists in the context, return the error message immediately
    if "error" in context:
        return context["error"]

    # Extract the relevant pieces of the user question context for formatting
    user_question_pre_text = context["pre_text"]
    user_question_table = format_table(context["table"])
    user_question_post_text = context["post_text"]

    # Construct the main question prompt with context, reasoning, and the
    # requested format
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
3. Please use the examples above "Let's think step by step" to do reasoning 
and calculation

4. Please produce the following outputs:
Logical Reasoning: similar to the Apply Logical Reasoning in the examples
Program: function-style operations or function call expressions
Answer: Just the final value as a string with a max of 2 digits decimal
Confidence: 0-100% certainty 

**Example Output Format:**
Do not add Analyse this question into output
Program: such as "multiply(2.12, const_1000), add(#0, 112), greater(#0, 5), 
Prefer subtract(x,const_100) over subtract(x,100)"
Answer: such as "5.2", "-4.9%", "8.92%", "$ 378.7 million", "2232", "no", "yes"
Confidence: 92% with percentage sign
Please organise the output in json format with the following keys: 
Logical Reasoning, Program, Answer, Confidence and with value as string.
       """

    # Combine all the example prompts and the final question prompt into one
    # final prompt
    final_prompt = "\n\n---\n\n".join(
        examples) + "\n\n---\n\n" + question_prompt

    # Return the final constructed prompt
    return final_prompt


# =============================================================================
# Step 2: Initialize the LLM
# =============================================================================
def query_gpt(prompt: str) -> str:
    """
    Sends a prompt to the GPT-3.5-turbo model and returns the generated
    response.
    """

    # Create a client instance to interact with the OpenAI API
    client = openai.OpenAI()  # This creates an OpenAI client object to send
    # requests to the API

    # Send the prompt to the GPT-3.5-turbo model, using the 'chat'
    # completion endpoint
    response = client.chat.completions.create(
        # Specify the model to use for generating the response
        model="gpt-3.5-turbo",

        # Provide the conversation context and the user input (the prompt)
        messages=[
            # The system message sets up the assistant's behavior and tone
            {"role": "system",
             "content": "You are a helpful financial analysis assistant."},

            # The user message contains the actual question or prompt that
            # GPT will respond to
            {"role": "user", "content": prompt}
        ],

        # Control the creativity of the response by adjusting the
        # temperature (0.0 for deterministic output)
        temperature=0.0)

    # Return the assistant's response, stripping any extra whitespace around
    # the content
    content = response.choices[0].message.content.strip()
    if not content:
        raise ValueError("Empty response from GPT. Please check the prompt.")
    return content


# =============================================================================
# Step 3: Generate Output
# ============================================================================
def generate_answer(question: str, num_example: int) -> str:
    """
    Generate the answer for a given question using preprocessed data.
    """
    # Ensure the data is loaded and preprocessed before inference
    if not shared_data.processed_dataset:
        raise ValueError("Data must be loaded and preprocessed first.")

    # Retrieve context for the given question
    context = query_data(question, shared_data.processed_dataset)
    # num_example = 3
    few_shot_prompt = generate_few_shot_prompt(shared_data.processed_dataset,
                                               question,
                                               context, num_example)
    # print("------ Prompt Sent to GPT ------\n", few_shot_prompt)
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
    program = context.get("program", "Program not found")
    answer = context.get("answer", "Answer not found")

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
