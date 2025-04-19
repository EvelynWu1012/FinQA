import os
import random
import csv
from .program_executor import execute_program
from src.prompt_LLM.prompt_answer_gen_inference import load_and_preprocess_data, \
    generate_answer
from src.shared import shared_data
from src.utils.utils import extract_llm_response_components, clean_text, is_numeric, \
    format_executable_answer


def exact_match_num(predicted: str, ground_truth: str,
                    threshold: float = 0.05) -> bool:
    """
    Checks if the predicted answer exactly matches the ground truth answer,
    comparing them as floating-point numbers rounded to two decimal places.
    If an exact match is not found, it checks for numeric proximity within
    the given threshold.

    Args:
        predicted (str): The predicted answer from the model -
                        it should be cleaned, can be converted to float
        ground_truth (str): The correct ground truth answer -
                        it should be cleaned, can be converted to float
        threshold (float): The numeric proximity threshold (default 0.05,
        i.e., 5%).

    Returns:
        bool: True if the exact match or numeric proximity match,
        False otherwise.
    """
    try:
        # Convert both predicted and ground_truth to floats and round to 2
        # decimal places
        pred_clean = round(float(predicted), 2)
        gt_clean = round(float(ground_truth), 2)
    except ValueError:
        return False  # If either value cannot be converted to float,
        # return False

    # Check if the values exactly match
    if pred_clean == gt_clean:
        return True

    # Check if the values are within the numeric proximity threshold
    return numeric_proximity(predicted, ground_truth, threshold)


def numeric_proximity(pred: str, truth: str, threshold=0.05) -> bool:
    """
    checks if the predicted value is within a specified proximity threshold
    of the truth value.

    Args:
        pred (str): The predicted answer.
        truth (str): The correct ground truth answer.
        threshold (float): The numeric proximity threshold (default 0.05,
        i.e., 5%).

    Returns:
        bool: True if the difference between the predicted and truth values
        is within the threshold.
    """
    try:
        pred_clean = round(float(pred), 2)
        gt_clean = round(float(truth), 2)
        return abs(pred_clean - gt_clean) / gt_clean <= threshold
    except ValueError:
        return False  # Return False if conversion fails


def exact_match_string(predicted, ground_truth):
    """
    checks whether two strings are exactly the same.

    Args:
        predicted (str): The predicted string.
        ground_truth (str): The ground truth string to compare against.

    Returns:
        bool: True if both strings are exactly equal, False otherwise.

    Example:
        exact_match_string("subtract(1, 2)", "subtract(1, 2)") -> True
        exact_match_string("subtract(1, 2)", "add(1, 2)") -> False
    """
    return predicted == ground_truth


def evaluate_answer_program(url: str, num_samples: int,
                         output_csv: str = "evaluation_results.csv",
                         seed: int = 42) -> tuple:
    """
    Evaluate the model on answer and program of randomly sampled questions
    and calculate accuracy.

    Args:
        url: Data source URL
        output_csv: Path to save evaluation results
        seed: Random seed for reproducibility
        num_samples: Number of samples to evaluate (default is 100)

    Returns:
        Accuracy percentage (0-100)
    """

    # Step 1: Check if data exists or load it
    processed_data = shared_data.processed_dataset
    if not processed_data:
        print("No preprocessed data found - loading now...")
        load_and_preprocess_data(url)

    # Step 2: Sample random questions
    # random.seed(seed)
    all_questions = list(processed_data.keys())
    if not processed_data:
        print("⚠️ No questions found in processed_data. Exiting evaluation.")
        return 0.0, None
    sample_size = min(num_samples, len(all_questions))
    sampled_questions = random.sample(all_questions, sample_size)

    # Step 3: Evaluation metrics
    results = []
    metrics = {
        'correct_answer': 0,
        'correct_program': 0,
        'total': 0,
    }

    for q in sampled_questions:
        metrics['total'] += 1
        metadata = processed_data[q]
        ground_truth_answer = str(metadata["answer"]).strip().lower()
        ground_truth_program = metadata["program"].strip().lower()
        ground_truth_exe_ans = format_executable_answer(metadata["exe_ans"])

        try:
            llm_output = str(generate_answer(q))
            parsed_output = extract_llm_response_components(llm_output)

            # check EXACT MATCH between predicted answer and ground truth answer
            prediction_answer_clean = clean_text(
                parsed_output.get("answer", "").strip().lower())
            # print("debug prediction_clean", prediction_answer_clean)
            ground_truth_answer_clean = clean_text(ground_truth_answer)
            if is_numeric(ground_truth_answer_clean) or is_numeric(
                    prediction_answer_clean):
                is_correct_answer = exact_match_num(prediction_answer_clean,
                                                    ground_truth_answer_clean)
            else:
                is_correct_answer = exact_match_string(prediction_answer_clean,
                                                       ground_truth_answer_clean)


            # check EXACT MATCH between predicted program and ground truth program
            prediction_program = parsed_output.get("program",
                                                   "").strip().lower()
            is_correct_program = exact_match_string(prediction_program,
                                                    ground_truth_program)

            # check Execution-Based Match between predicted program and ground truth program
            try:
                prediction_exec_result = execute_program(prediction_program, q)

                if is_numeric(prediction_exec_result) or is_numeric(ground_truth_exe_ans):
                    is_program_calc_match = exact_match_num(prediction_exec_result, ground_truth_exe_ans)
                else:
                    is_program_calc_match = exact_match_string(prediction_exec_result,ground_truth_exe_ans)
            except Exception as exec_err:
                print(f"Execution error for question: {q[:60]}... -> {exec_err}")
                is_program_calc_match = False

            # save the results
            results.append({
                "question": q[:200],
                "prediction_answer": prediction_answer_clean,
                "ground_truth_answer": ground_truth_answer_clean,
                "exact_match_answer": bool(is_correct_answer),
                "prediction_program": prediction_program,
                "ground_truth_program": ground_truth_program,
                "exact_match_program": bool(is_correct_program),
                "calculation_match_program": bool(is_program_calc_match)
            })

            metrics['correct_answer'] += is_correct_answer
            metrics['correct_program'] += is_program_calc_match

        except Exception as e:
            results.append({
                "question": q[:200],
                "prediction": f"ERROR: {str(e)}",
                "ground_truth_answer": ground_truth_answer.strip().lower(),
                "ground_truth_program": ground_truth_program.strip().lower(),
                "exact_match": False
            })

    # Step 4: Save results
    if results:
        # Create results directory in current working directory
        results_dir = "../../results"  # This will create it in current working directory
        # exist_ok=True prevents errors if folder exists
        os.makedirs(results_dir, exist_ok=True)
        # exist_ok=True prevents errors if folder exists
        os.makedirs(results_dir, exist_ok=True)
        # Define output path
        output_path = os.path.join(results_dir, output_csv)
        with open(output_path, mode="w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"✅ Results successfully saved to {output_path}")
    else:
        print("⚠️ No results to write to CSV.")

    # Step 5: Display metrics
    accuracy_answer = (metrics['correct_answer'] / metrics['total']) * 100 if \
        metrics['total'] else 0
    accuracy_program = (metrics['correct_program'] / metrics['total']) * 100 \
        if metrics['total'] else 0

    print("\n📊 Evaluation Summary:")
    print(f"• Exact Matches Answer: {metrics['correct_answer']}/{metrics['total']}")
    print(f"• Accuracy Answer: {accuracy_answer:.2f}%")
    print(f"• Exact Matches(EM) Program {metrics['correct_program']}/{metrics['total']}")
    print(f"• Accuracy Program: {accuracy_program:.2f}%")
    print(f"• Results saved to: {output_csv}")

    return accuracy_answer, accuracy_program

