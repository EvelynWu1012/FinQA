# prompt_program_gen.py

import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv
import openai
import os
import executor  # Importing executor.py to use load_data function

# SETUP
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ----------- 1. Build a Retriever (RAG style) -----------
"""
Purpose: Find the most relevant rows or paragraphs from a table or document 
based on a question.
"""


def load_table_context():
    """
    Loads the data using load_data from executor.py, retrieves necessary
    fields,
    splits the table into chunks, and embeds them for further processing.

    Returns:
    - db (FAISS): FAISS database with embedded table chunks.
    """
    # Step 1: Load the data using load_data function from executor.py
    url = "your_data_url_here"  # Replace with actual URL
    zip_file_path = "data.zip"
    extract_to = "data"
    json_file = "train.json"
    data = executor.load_data(url, zip_file_path, extract_to, json_file)

    # Step 2: Process the loaded data
    context = []
    for item in data:
        # Retrieve necessary fields from the data
        pre_text = item.get("pre_text", "")
        post_text = item.get("post_text", "")
        table_text = item.get("table", "")

        # Get the QA fields
        qa = item.get("qa", {})
        question = qa.get("question", "")
        answer = qa.get("answer", "")
        ann_table_rows = qa.get("ann_table_rows", [])
        steps = qa.get("steps", [])
        program = qa.get("program", "")
        exe_ans = qa.get("exe_ans", "")

        # Concatenate the fields to form the context
        full_text = (f"{pre_text}\n{post_text}\n{table_text}\n{question}\n"
                     f"{answer}\n")
        # Assuming ann_table_rows are a list of rows
        full_text += "\n".join([str(row) for row in
                                ann_table_rows])
        full_text += "\n".join([str(step) for step in
                                steps])  # Assuming steps are a list of steps
        full_text += f"\nProgram: {program}\nExecution Answer: {exe_ans}"

        # Add the full text to context
        context.append(full_text)

    # Step 3: Split text into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,
                                                   chunk_overlap=10)
    texts = text_splitter.split_text("\n".join(context))

    # Step 4: Create embeddings and FAISS index
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(texts, embeddings)

    return db


# ----------- 2. Prompt Template for Program Reasoning -----------
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Context:
{context}

Question: {question}

Based on the above context, provide a reasoning chain as a list of steps, 
and then write the final program using the following functions:
- subtract(x, y)
- divide(x, y)

Use intermediate variables like #0, #1 to represent results from earlier steps.

Format:
Steps:
1. ...
2. ...
Program: ...
"""
)

# ----------- 3. LLMChain Wrapper -----------
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
chain = LLMChain(llm=llm, prompt=prompt_template)


# ----------- 4. Run on Example QA -----------
def run_chain_on_example(table_text, question):
    retriever = load_table_context(table_text)
    docs = retriever.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    result = chain.run(context=context, question=question)
    return result


# ----------- 5. Example Run -----------
if __name__ == "__main__":
    table_text = """
    | Year | Net cash from operating activities |
    |------|------------------------------------|
    | 2008 | 181001                             |
    | 2009 | 206588                             |
    """

    question = ("What was the percentage change in the net cash from "
                "operating activities from 2008 to 2009?")

    output = run_chain_on_example(table_text, question)
    print("==== OUTPUT ====")
    print(output)
