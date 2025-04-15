"""

1. Load and Preprocess JSON Data
2. Split Documents and Embed
3. Use a Vector Store (FAISS) for Retrieval
4. Use LangChain’s RAG + ReAct agent to run reasoning over context
5. Answer the question using GPT-3.5
"""

import json
from typing import List, Dict, Any
from langchain.schema import Document
from executor import load_data

def build_documents_from_json(url: str, zip_file_path: str = "data.zip",
                              extract_to: str = "data", json_file: str = "train.json") -> List[Document]:
    from executor import load_data

    data = load_data(url, zip_file_path, extract_to, json_file)
    if isinstance(data, dict):
        data = [data]

    documents = []

    for entry in data:
        text = '\n'.join(entry.get('pre_text', [])) + '\n' + '\n'.join(entry.get('post_text', []))
        table_data = '\n'.join([" | ".join(row) for row in entry.get("table", [])])
        full_text = f"{text}\nTable:\n{table_data}"

        # Add QA section
        qa = entry.get("qa", {})
        if qa:
            full_text += f"\n\nQuestion: {qa.get('question', '')}\nAnswer: {qa.get('answer', '')}"
            steps = qa.get('steps', [])
            for step in steps:
                full_text += f"\n{step['op']}({step['arg1']}, {step['arg2']}) = {step['res']}"

        # Add annotations
        annotation = entry.get("annotation", {})
        if annotation:
            if "step_list" in annotation:
                full_text += "\n\nStep-by-step:\n" + '\n'.join(annotation["step_list"])
            for key in ["answer_list", "dialogue_break", "turn_program_ori",
                        "dialogue_break_ori", "turn_program", "qa_split", "exe_ans_list"]:
                if key in annotation:
                    full_text += f"\n\n{key}:\n{annotation[key]}"

        metadata = {
            "source": entry.get("filename", "unknown"),
            "question": qa.get('question', ''),
            "ops": [s["op"] for s in qa.get("steps", [])] if qa.get("steps") else [],
        }

        documents.append(Document(page_content=full_text, metadata=metadata))

    return documents


url = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"
documents = build_documents_from_json(url)

# Print the details for each document
for doc in documents[:5]:
    print("Page Content Preview:")
    lines = doc.page_content.split('\n')
    print(f"(pre_text preview...) {lines[0]}")
    print(f"(post_text preview...) {lines[-1]}")

    print("\nTable:")
    for line in lines:
        if " | " in line:
            print(line)

    for line in lines:
        if line.startswith("Question:") or line.startswith("Answer:"):
            print(line)

    print("\nQA Reasoning Steps:")
    for line in lines:
        if " = " in line:
            print(line)

    print("\nAnnotations (if any):")
    for key in ["step_list", "answer_list", "dialogue_break", "turn_program_ori",
                "dialogue_break_ori", "turn_program", "qa_split", "exe_ans_list"]:
        if any(key in line for line in lines):
            print(f"{key} present")

    print("\n" + "-" * 40)
