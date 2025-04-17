import random
import csv
from prompt_answer_gen_inference import load_and_preprocess_data, generate_answer
import shared_data
from utils import extract_llm_response_components


def exact_match(predicted: str, ground_truth: str) -> int:
    """
    Checks if the predicted answer exactly matches the ground truth answer.

    Args:
        predicted (str): The predicted answer from the model.
        ground_truth (str): The correct ground truth answer.

    Returns:
        int: 1 if exact match, 0 otherwise.
    """
    # Normalize both strings by stripping whitespace and converting to
    # lowercase
    pred_clean = predicted.strip().lower()
    gt_clean = ground_truth.strip().lower()

    return int(pred_clean == gt_clean)


def evaluate_answer_match(url: str, num_samples: int, output_csv: str = "test_results.csv",
                       seed: int = 42) -> float:
    """
    Evaluate the model on a specified number of randomly sampled questions
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
    print(
        f"DEBUG: processed_data retrieved with {len(processed_data)} examples.")
    if not processed_data:
        print("No preprocessed data found - loading now...")
        load_and_preprocess_data(url)

    # Step 2: Sample random questions
    random.seed(seed)
    all_questions = list(processed_data.keys())
    if not processed_data:
        print("⚠️ No questions found in processed_data. Exiting evaluation.")
        return 0.0
    sample_size = min(num_samples, len(all_questions))
    sampled_questions = random.sample(all_questions, sample_size)

    # Step 3: Evaluation metrics
    results = []
    metrics = {
        'correct': 0,
        'errors': 0,
        'total': 0,
        'close_matches': 0
    }

    for q in sampled_questions:
        metrics['total'] += 1
        metadata = processed_data[q]
        ground_truth = str(metadata["answer"])

        try:
            llm_output = str(generate_answer(q))
            print("debug llm_output", llm_output)
            parsed_output = extract_llm_response_components(llm_output)
            print("debug parsed_output", parsed_output)
            prediction = parsed_output.get("answer", "").strip().lower()
            print("debug prediction", prediction)
            gt_clean = ground_truth.strip().lower()
            print("debug groud_truth", gt_clean)
            is_correct = exact_match(prediction, ground_truth)

            results.append({
                "question": q[:200],
                "prediction": prediction,
                "ground_truth": gt_clean,
                "exact_match": bool(is_correct)
            })

            metrics['correct'] += is_correct

        except Exception as e:
            metrics['errors'] += 1
            results.append({
                "question": q[:200],
                "prediction": f"ERROR: {str(e)}",
                "ground_truth": ground_truth.strip().lower(),
                "exact_match": False
            })

    # Step 4: Save results
    if results:
        with open(output_csv, mode="w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    else:
        print("⚠️ No results to write to CSV.")

    # Step 5: Display metrics
    accuracy = (metrics['correct'] / metrics['total']) * 100 if metrics['total'] else 0

    print("\n📊 Evaluation Summary:")
    print(f"• Exact Matches: {metrics['correct']}/{metrics['total']}")
    print(f"• Accuracy: {accuracy:.2f}%")
    print(f"• Errors: {metrics['errors']}")
    print(f"• Results saved to: {output_csv}")

    return accuracy


