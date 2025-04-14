"""
Goal: Evaluate executor on all validation samples.
Tasks:
- Run `program` for all `train.json` and `dev.json`.
- Calculate execution accuracy (i.e., how many correct numeric answers are produced).
- Handle errors gracefully (invalid ops, missing values).
- Validates the program executor in Option B: FinQA-style Program Execution is working.
- Gives you a reliable “ground truth” function for checking GPT-generated programs.
- Use logs to improve prompts and models.
"""