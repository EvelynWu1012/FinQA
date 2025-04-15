"""
1. Load and Preprocess JSON Data
2. Split Documents and Embed
3. Use a Vector Store (FAISS) for Retrieval
4. Use LangChain's RAG + ReAct agent to run reasoning over context
5. Answer the question using GPT-3.5
"""

import json
from typing import List, Dict, Any
from langchain.schema import Document


def build_documents_from_json(url: str, zip_file_path: str = "data.zip",
                              extract_to: str = "data",
                              json_file: str = "train.json") -> List[Document]:
    # Import inside function to prevent immediate execution when module is imported
    from executor import load_data

    data = load_data(url, zip_file_path, extract_to, json_file)
    if isinstance(data, dict):
        data = [data]

    documents = []

    for entry in data:
        # Handle pre_text and post_text properly
        pre_text = 'PRE_TEXT: ' + '\n'.join(
            entry.get('pre_text', [])) if entry.get('pre_text') else ''
        post_text = 'POST_TEXT: ' + '\n'.join(
            entry.get('post_text', [])) if entry.get('post_text') else ''
        text = f"{pre_text}\n{post_text}" if pre_text or post_text else ''

        table_data = '\n'.join(
            [" | ".join(row) for row in entry.get("table", [])])
        full_text = f"{text}\nTable:\n{table_data}" if text or table_data else ''

        # Add QA section
        qa = entry.get("qa", {})
        if qa:
            full_text += f"\n\nQuestion: {qa.get('question', '')}\nAnswer: {qa.get('answer', '')}"
            steps = qa.get('steps', [])
            for step in steps:
                full_text += f"\n{step['op']}({step['arg1']}, {step['arg2']}) = {step['res']}"

        # Add annotations with full content
        annotation = entry.get("annotation", {})
        if annotation:
            if "step_list" in annotation:
                full_text += "\n\nStep-by-step:\n" + '\n'.join(
                    annotation["step_list"])
            for key in ["answer_list", "dialogue_break", "turn_program_ori",
                        "dialogue_break_ori", "turn_program", "qa_split",
                        "exe_ans_list"]:
                if key in annotation:
                    full_text += f"\n\n{key}:\n{json.dumps(annotation[key], indent=2)}"

        metadata = {
            "source": entry.get("filename", "unknown"),
            "question": qa.get('question', ''),
            "ops": [s["op"] for s in qa.get("steps", [])] if qa.get(
                "steps") else [],
        }

        documents.append(Document(page_content=full_text, metadata=metadata))

    return documents


if __name__ == "__main__":
    # This block will only run when executing this script directly
    url = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"
    documents = build_documents_from_json(url)
    print(documents[0])
