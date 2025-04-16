import json
from dotenv import load_dotenv
import os
import zipfile
from typing import Dict, List
import requests
import openai
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import OpenAI old
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

# Global variable to hold the max_samples value
MAX_SAMPLES = 3037


# =============================================================================
# Step 1: Preprocess Examples
# =============================================================================
def load_data(url: str, zip_file_path: str = "data.zip",
              extract_to: str = "data", json_file: str = "train.json"):
    """
    Downloads a zip file from the given URL, extracts it, and loads the
    specified JSON file.

    Args:
    - url (str): URL to download the zip file from.
    - zip_file_path (str): Path where the zip file will be saved.
    - extract_to (str): Directory to extract the contents of the zip file.
    - json_file (str): The name of the JSON file to load from the extracted
    folder.

    Returns:
    - data (dict): The data loaded from the specified JSON file.
    """
    # 1. Download the zip file
    print(f"Downloading {zip_file_path} from {url}...")
    # Send a GET request to the provided URL
    response = requests.get(url)
    # Open the specified path to save the zip file in write-binary mode.
    with open(zip_file_path, "wb") as f:
        # Write the content of the response(the downloaded zip file) to the
        # local file
        f.write(response.content)
    print(f"Download complete: {zip_file_path}")

    # 2. Unzip the file
    print(f"Unzipping {zip_file_path} into {extract_to}...")
    # Open the zip file in read mode
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        # Extract all the contents of the zip file into the specified directory
        zip_ref.extractall(extract_to)
    print(f"Unzip complete. Files extracted to: {extract_to}")

    # 3. Load the JSON file
    # Create the full path to the JSON file inside the extracted folder
    json_path = os.path.join(extract_to, "data", json_file)
    print(f"Loading JSON data from {json_path}...")
    with open(json_path, "r") as f:  # Open the JSON file in read mode
        # Load and parse the JSON data from the file into a Python dict
        data = json.load(f)
    print(f"Loaded {len(data)} examples from {json_file}.")

    return data


def preprocess_example(example: Dict) -> Dict:
    """
    For a single example, return a dictionary with the question as the key and
    associated metadata as the value.
    """
    # Find the first 'qa' variant key (e.g., 'qa', 'qa_0', 'qa_1', etc.)
    qa_key = next((key for key in example.keys() if key.startswith("qa")),
                  None)

    if not qa_key:
        raise KeyError("No 'qa' key found in the example.")

    # Extract information from the found 'qa' key
    question = example[qa_key]["question"]
    table = example["table"]
    table_header = table[0]
    ann_table_rows = example[qa_key].get("ann_table_rows", [])

    # Extract just the annotated rows for simplicity
    focused_rows_header = [table_header] + [table[i] for i in ann_table_rows]

    return {
        question: {
            "table": table,
            "focused_rows": focused_rows_header,
            "steps": example[qa_key].get("steps", []),
            "program": example[qa_key].get("program", ""),
            "exe_ans": example[qa_key].get("exe_ans"),
            "answer": example[qa_key].get("answer"),
        }
    }


def preprocess_dataset(data: List[Dict], max_samples) -> Dict:
    """
    Preprocess the entire dataset and return a dictionary where each key is
    a question
    and the corresponding value is the associated metadata.

    Args:
    - data (List[Dict]): The list of data examples.
    - max_samples (int): The maximum number of samples to process.

    Returns:
    - Dict: A dictionary with questions as keys and their corresponding
    metadata as values.
    """
    # Initialize an empty dictionary to hold the results
    processed_data = {}

    # Process each example
    for example in data[:max_samples]:
        # Get the preprocessed data for the example
        example_data = preprocess_example(example)

        # Since the key in the result is the question, we'll update the main
        # dictionary with the question as key
        processed_data.update(example_data)

    return processed_data


# =============================================================================
# Step 2: Set up LangChain Prompt Template
# =============================================================================
# Initialize OpenAI API Key
load_dotenv()  # This loads variables from .env into the environment
openai.api_key = os.getenv("OPENAI_API_KEY")
print(f"Loaded API key: {openai.api_key[:5]}...")  # Don’t print the full key!


# Define a function for querying the processed data
def query_data(question: str, processed_data: Dict) -> str:
    """
    Given a question, return the associated context from the preprocessed data.
    """
    return processed_data.get(question, "Data for this question not found.")


# Step 3: Create a LangChain Prompt Template
# Define a prompt template to generate an answer or code based on the
# question and context.
prompt_template = PromptTemplate(
    input_variables=["question", "data"],
    template="""
    You are a helpful assistant capable of reasoning through data. Given a 
    question about an investment or finance,
    and the associated data from a table, answer the question or generate a 
    program to solve it.

    Here is the question:
    {question}

    Here is the data:
    {data}

    Please provide the answer or the program:
    """
)

# =============================================================================
# Step 3: Initialize the LLM
# =============================================================================
# Initialize the OpenAI LLM (GPT-4)
# llm = OpenAI(model="gpt-3.5-turbo")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# New-style chain using RunnableSequence
chain = prompt_template | llm | StrOutputParser()


# =============================================================================
# Step 4: Generate Output
# ============================================================================
def generate_output(question: str, processed_data: Dict):
    """
    This function generates a response using the LangChain agent.
    It takes a question and the processed data as input.
    """
    result = agent.invoke(question)
    return result


if __name__ == "__main__":
    # This block will only run when executing this script directly
    url = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"
    data = load_data(url)
    processed = preprocess_dataset(data, 1)
    print("Processed Data:", processed)
    # Step 2: Set up tools using the processed data
    tools = [
        Tool(
            name="Preprocessed Data Query",
            func=lambda question: query_data(question, processed),
            description="This tool allows querying the preprocessed dataset based on the question."
        )
    ]

    # Step 3: Initialize the agent
    agent = initialize_agent(
        tools,
        llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )

    # Step 4: Ask a question
    question_text = "what was the percentage change in the net cash from operating activities from 2008 to 2009"
    print("Asking question:", question_text)
    print("Processed Data for this question:", processed.get(question_text))
    print("Tools available to the agent:", tools)
    print("Agent plan:")
    print("Answer:", agent.invoke(question_text))

    # example = data[3000]  # Get the first example
    # print(preprocess_example(example))
    # Print out the preprocessed data for a specific question
    #question_text = ("what is the roi of an investment in ups in 2004 and "
                     #"sold in 2006?")
    #result = processed.get(question_text)

    #if result:
        #print("Found the question in the dataset:", result)
    #else:
        #print("Question not found in the dataset.")
