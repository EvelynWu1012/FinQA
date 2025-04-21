# FinQA

The FinQA system is designed to support complex financial question answering by integrating data processing, intelligent prompt construction, and large language model inference within a modular architecture. It orchestrates each stage—from raw data ingestion to model evaluation—through clearly separated components that promote reliability, maintainability, and performance. By leveraging semantic similarity search and efficient pipeline design, the system ensures high-quality, contextually relevant responses tailored to financial domains.

## Features

- **Data Loader**: downloads, extracts, and validates datasets.
- **Preprocessing**: prepares the input data for downstream tasks by extracting and structuring relevant information, ensuring consistency and relevance for later inference stages
- **Prompt Engineering and LLM Inference**: dynamically selects the most relevant examples for a given input using FAISS similarity search to enhance the quality of prompts and implements prompt engineering logic and generates model responses using the LLM
- **Evaluation Package:**: parses and evaluates model-generated answers and intermediate programs using relevant metrics. It encapsulates evaluation logic separately from other stages of the pipeline
- **Shared Package**: provides reusable components for caching, FAISS index handling, and other shared tasks across the system.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/EvelynWu1012/FinQA
   cd FinQA

2. Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install dependencies:
   ```bash
    pip install -r requirements.txt

# Usage


Run the main script:
    
    python main.py

# Key Functionalities:  

*   Load or preprocess datasets.
*   Initialize or load FAISS index.
*   Generate answers and evaluate them.

## Example Question:  
What was the percent of the growth in the revenues from 2007 to 2008?



# Technologies Used

*   Python: Core programming language.
*   FAISS: For similarity search and indexing.
*   Pytest: For testing the project.
*   LLM: OpenAI 3.5 turbo
*   SentenceTransformer: all-MiniLM-L6-v2


# Contributing
Contributions are welcome! Please follow these steps:  
Fork the repository.
Create a new branch for your feature or bug fix.
Submit a pull request with a detailed description of your changes.
# License
This project is licensed under the MIT License. See the LICENSE file for details.  
# Contact
For questions or feedback, please contact evelyn.qianqian@gmail.com.