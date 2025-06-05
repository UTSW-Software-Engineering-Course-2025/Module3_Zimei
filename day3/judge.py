# get a llm judge to read it out from the file to judge the prediction and the truth 

import os
from dotenv import load_dotenv
import pandas as pd  # Ensure pandas is imported
from tqdm import tqdm 
import matplotlib.pyplot as plt
from collections import defaultdict
from openai import AzureOpenAI
import re
import mlflow
DATASET = "genehop"


# In[13]:

os.environ['no_proxy'] = '*'
mlflow.set_tracking_uri("http://198.215.61.34:8153/")
mlflow.set_experiment("s440797-Zimei")
# 2.2 Model Configuration
# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API client
# NOTE: HERE we dont ever save the API key as a variable so it never gets printed
api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
api_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

DATASET = "genehop"

DATASET_FILE = f"{DATASET}_results.json"
# set up the client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  # Always use it just in time
    api_version=api_version,
    azure_endpoint=api_base,
)
# --- LLM Judge Function ---
def evaluate_prediction(question: str, prediction: str, ground_truth: str, rubric: str) -> str:
    """
    Uses an LLM to evaluate a prediction against a ground truth answer based on a rubric.

    Args:
        question: The question that was asked.
        prediction: The model's generated answer.
        ground_truth: The correct or "gold standard" answer.
        rubric: The criteria for judging the prediction.

    Returns:
        A string containing the score and explanation from the LLM judge.
    """
    prompt = f"""You are a domain expert in genetics and biology, tasked with evaluating an AI's response.

**Question:**
{question}

**AI's Prediction:**
{prediction}

**Ground Truth Answer:**
{ground_truth}

**Evaluation Rubric:**
{rubric}

**Instructions:**
Please score the AI's prediction on a scale of 0 to 1 based on the provided rubric. Provide a clear and concise justification for your score, referencing the ground truth. Your output must be in the following format:

Score: [Your Score]
Explanation: [Your Justification]
"""
    try:
        response = client.chat.completions.create(
            model=api_deployment,
            messages=[
                {"role": "system", "content": "You are a meticulous evaluator, strictly following the provided rubric to assess the accuracy and completeness of biomedical answers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred during evaluation: {e}"

def score_per_task_all(results):
    task_scores = defaultdict(list)
    for result in results:
        if result.success:
            task_scores[result.task].append(result.score)
    for task, scores in task_scores.items():
        task_scores[task] = sum(scores) / len(scores) if scores else 0
    all_scores = [result.score for result in results if result.success]
    overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return task_scores,overall_score


def plot_scores_by_task(
    task_scores: dict[str, str], overall_score: float, imag_path: str
) -> None:
    tasks = list(task_scores.keys())
    scores = list(task_scores.values())

    plt.figure(figsize=(10, 6))
    plt.barh(tasks, scores, color="skyblue")  # Changed from barh to bar
    plt.axvline(
        overall_score,
        color="red",
        linestyle="--",
        label=f"Overall Score={overall_score}",
    )  # Changed from axvline to axhline
    plt.xlabel("Score")  # Changed xlabel to ylabel
    plt.title("Scores by Task")
    plt.legend()
    plt.grid(axis="x")  # Changed from axis='x' to axis='y'
    plt.savefig(imag_path)
    plt.show()

# def calculate_evaluation(results_df):
#     """
#     Calculates the average score for each task and the overall average score.

#     Args:
#         results_df: A pandas DataFrame containing a 'task' and a'evaluation' column.

#     Returns:
#         A tuple containing:
#         - task_scores (dict): Average score for each task.
#         - overall_score (float): Average score across all entries.
#     """
#     results_df['evaluation'] = pd.to_numeric(results_df['evaluation'], errors='coerce')
    
#     # Drop rows where score could not be parsed
#     results_df.dropna(subset=['evaluation'], inplace=True)

#     task_scores = results_df.groupby('task')['evaluation'].mean().to_dict()
#     overall_score = results_df['evaluation'].mean()
    
#     return task_scores, overall_score

# --- Main Execution ---
if __name__ == "__main__":
    # --- Define the Rubric for the "genehop" dataset ---
    # This rubric is tailored for evaluating answers in a biomedical context.
    genehop_rubric = """
    **1: Incorrect or Irrelevant:** The prediction is factually incorrect, nonsensical, or completely irrelevant to the question.
    **2: Partially Correct but Incomplete:** The prediction contains some correct information but misses significant details or context present in the ground truth. It may contain minor inaccuracies.
    **3: Mostly Correct:** The prediction is factually correct and answers the main point of the question but lacks the full completeness, depth, or nuance of the ground truth.
    **4: Correct and Complete:** The prediction is factually accurate, comprehensive, and aligns fully with the ground truth answer.
    **5: Exceptional (Correct, Complete, and Clear):** The prediction is not only correct and complete but is also exceptionally well-explained, clear, and easy to understand, potentially offering valuable context not explicitly in the ground truth but still highly relevant.
    """

    # --- Load the Dataset ---
    try:
        results_df = pd.read_json(DATASET_FILE, lines=True)
        # This assumes your JSON file has columns named 'question', 'prediction', and 'answer'.
        # If your column names are different, please modify the lines below.
        # For example, if your prediction column is 'pred', change 'prediction' to 'pred'.
        required_columns = ['question', 'processed_prediction', 'answer']
        if not all(col in results_df.columns for col in required_columns):
            print(f"Error: The JSON file must contain the columns: {required_columns}")
            exit()

    except FileNotFoundError:
        print(f"Error: The file '{DATASET_FILE}' was not found.")
        exit()
    except Exception as e:
        print(f"An error occurred while reading the JSON file: {e}")
        exit()

    # --- Iterate and Evaluate Each Entry ---
    evaluations = []
    print(f"Starting evaluation of {len(results_df)} predictions from '{DATASET_FILE}'...")


    # Start MLflow run
    with mlflow.start_run(run_name="llm_judge"):
        evaluations = []
        scores = []
        print(f"Starting evaluation of {len(results_df)} predictions from '{DATASET_FILE}'...")

        for index, row in results_df.iterrows():
            evaluation_result = evaluate_prediction(
                question=row['question'],
                prediction=row['processed_prediction'],
                ground_truth=row['answer'],
                rubric=genehop_rubric
            )
            evaluations.append(evaluation_result)

            # Use regex to robustly extract the score
            score_match = re.search(r"Score:\s*(\d+(\.\d+)?)", evaluation_result)
            if score_match:
                scores.append(float(score_match.group(1)))
            else:
                scores.append(None) # Append None if score is not found

        # Add new columns to the DataFrame
        results_df['llm_evaluation'] = evaluations
        results_df['score'] = scores

        # --- Save the Evaluated Results to a New JSON File ---
        output_file = f"{DATASET}_evaluated.json"
        results_df.to_json(output_file, orient='records', indent=4)
        print(f"Evaluations successfully saved to '{output_file}'")

        # --- Calculate and Plot Scores ---
        task_scores, overall_score = score_per_task_all(results_df)
        imag_path =f"{DATASET}_scores_by_task_llm_judge.png"
        plot_scores_by_task(task_scores, overall_score,imag_path )
        mlflow.log_artifact(imag_path)
        mlflow.log_metric("overall_score", overall_score)
            # Log the scores by task
        for task, score in task_scores.items():
            mlflow.log_metric(f"score_{task}", score)
        print("\n--- Evaluation Score Summary ---")
        if task_scores:
            for task, score in task_scores.items():
                print(f"  - Task: {task}, Average Score: {score:.2f}")
            print(f"------------------------------------")
            print(f"  - Overall Average Score: {overall_score:.2f}")

            # --- Generate and Display the Plot ---
            plot_scores_by_task(task_scores, overall_score, f"{DATASET}_scores_by_task.png")
        else:
            print("No valid scores were found to generate a plot.")

        mlflow.log_artifact(output_file)
