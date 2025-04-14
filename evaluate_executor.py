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
import math
import os
from typing import Dict, List, Tuple
import pandas as pd
from executor import load_data, parse_table, execute_program


class ExecutorEvaluator:
    def __init__(self):
        self.results = []
        self.error_logs = []
        self.metrics = {
            'total_samples': 0,
            'successful_executions': 0,
            'correct_answers': 0,
            'error_types': {}
        }

    def evaluate_dataset(self, data: List[Dict], dataset_name: str = "train") -> Dict:
        """
        Evaluate the executor on an entire dataset (train or dev)
        """
        print(f"\nEvaluating on {dataset_name} dataset ({len(data)} examples)...")
        # for example in tqdm(data):  # Optional: Add a progress bar
        for example in data:
            table = parse_table(example["table"])
            qa_pairs = self._extract_qa_pairs(example)

            for qa in qa_pairs:
                self.metrics['total_samples'] += 1
                self._evaluate_single_qa(qa, table, example.get('id', 'unknown'), dataset_name)

        return self._compute_metrics(dataset_name)

    def _extract_qa_pairs(self, example: Dict) -> List[Dict]:
        """
        Extract QA pairs from a data example in a flexible way.
        """
        qa_pairs = []
        i = 0
        while True:
            qa_key = f"qa_{i}"
            if qa_key in example:
                qa_pairs.append(example[qa_key])
                i += 1
            else:
                if i == 0 and "qa" in example:
                    qa_pairs.append(example["qa"])
                break
        return qa_pairs

    def _evaluate_single_qa(self, qa: Dict, table: List[Dict], example_id: str, dataset_name: str):
        """
        Evaluate a single QA pair and record results
        """
        try:
            program = qa["program"]
            question = qa["question"]

            # Handle percentage answers robustly
            answer_str = qa["exe_ans"]
            expected_answer = self._parse_answer(answer_str)

            try:
                predicted_answer = execute_program(program, table)
                # Use math.isclose() to check if answers are "close enough"
                tolerance = 1e-2  # Adjust this tolerance based on required precision
                if isinstance(predicted_answer, (int, float)) and isinstance(
                        expected_answer, (int, float)):
                    is_correct = math.isclose(predicted_answer,
                                              expected_answer,
                                              abs_tol=tolerance)
                else:
                    is_correct = str(predicted_answer).strip().lower() == str(
                        expected_answer).strip().lower()

                self.metrics['successful_executions'] += 1
                if is_correct:
                    self.metrics['correct_answers'] += 1

                self.results.append({
                    'dataset': dataset_name,
                    'example_id': example_id,
                    'question': question,
                    'program': program,
                    'expected_answer': expected_answer,
                    'predicted_answer': predicted_answer,
                    'is_correct': is_correct,
                    'error_type': None
                })

            except Exception as e:
                self._record_error(
                    dataset_name, example_id, question, program,
                    expected_answer, str(e), type(e).__name__
                )

        except KeyError as e:
            self._record_error(
                dataset_name, example_id, qa.get('question', ''),
                program=None, expected_answer=None,
                error=str(e), error_type="MissingKey"
            )

    def _parse_answer(self, answer_str) -> float:
        """
        Parse the expected answer from string (handling percentages and non-numeric values).
        """
        if isinstance(answer_str, str):
            # Remove commas and lowercase
            stripped = answer_str.strip().replace(",", "").lower()
            try:
                if stripped.endswith("%"):
                    # Handle percentage
                    return float(stripped.rstrip("%")) / 100
                return float(stripped)
            except ValueError:
                # Return the original string if it's not convertible to float
                return stripped
        try:
            return float(answer_str)
        except ValueError:
            return str(answer_str).strip().lower()

    def _record_error(
        self, dataset: str, example_id: str, question: str,
        program, expected_answer, error: str, error_type: str
    ):
        """
        Helper to log and count an error
        """
        self.metrics['error_types'][error_type] = self.metrics['error_types'].get(error_type, 0) + 1

        self.results.append({
            'dataset': dataset,
            'example_id': example_id,
            'question': question,
            'program': program,
            'expected_answer': expected_answer,
            'predicted_answer': None,
            'is_correct': False,
            'error_type': error_type
        })

        self.error_logs.append({
            'dataset': dataset,
            'example_id': example_id,
            'question': question,
            'program': program,
            'error': error,
            'error_type': error_type
        })

    def _compute_metrics(self, dataset_name: str) -> Dict:
        """
        Compute evaluation metrics for the dataset
        """
        total = self.metrics['total_samples']
        executed = self.metrics['successful_executions']
        correct = self.metrics['correct_answers']
        if total == 0:
            return {}

        return {
            'dataset': dataset_name,
            'total_samples': total,
            'execution_accuracy': correct / total,
            'coverage': executed / total,
            'correct_of_executed': correct / executed if executed else 0,
            'error_types': self.metrics['error_types']
        }

    def save_results(self, output_dir: str = "results"):
        """
        Save evaluation results to CSV files
        """
        os.makedirs(output_dir, exist_ok=True)

        results_df = pd.DataFrame(self.results)
        results_path = os.path.join(output_dir, "executor_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Saved detailed results to {results_path}")

        if self.error_logs:
            errors_df = pd.DataFrame(self.error_logs)
            errors_path = os.path.join(output_dir, "executor_errors.csv")
            errors_df.to_csv(errors_path, index=False)
            print(f"Saved error logs to {errors_path}")


def main():
    evaluator = ExecutorEvaluator()

    url = "https://github.com/czyssrs/ConvFinQA/raw/main/data.zip"
    data_dir = "data"
    zip_file_path = "data.zip"

    train_data = load_data(url, zip_file_path, data_dir, "train.json")
    dev_data = load_data(url, zip_file_path, data_dir, "dev.json")

    train_metrics = evaluator.evaluate_dataset(train_data, "train")
    dev_metrics = evaluator.evaluate_dataset(dev_data, "dev")

    print("\n=== Evaluation Summary ===")
    for metrics in [train_metrics, dev_metrics]:
        print(f"\nDataset: {metrics['dataset']}")
        print(f"Total samples: {metrics['total_samples']}")
        print(f"Execution accuracy: {metrics['execution_accuracy']:.2%}")
        print(f"Coverage: {metrics['coverage']:.2%}")
        print(f"Correct when executed: {metrics['correct_of_executed']:.2%}")
        print("Error types:")
        for error, count in metrics['error_types'].items():
            print(f"  {error}: {count} ({count / metrics['total_samples']:.2%})")

    evaluator.save_results()


if __name__ == "__main__":
    main()
