#!/usr/bin/env python
# coding: utf-8

# ## 1. Imports

# In[1]:


import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass

# 1.1 Place imports here
from typing import List, Optional

import matplotlib.pyplot as plt
import mlflow
import pandas as pd  # Ensure pandas is imported
from dotenv import load_dotenv
from openai import AzureOpenAI
from tqdm import tqdm

# ## 2. Configuration

# In[2]:

os.environ["no_proxy"] = "*"
mlflow.set_tracking_uri("http://198.215.61.34:8153/")
# os.chdir("/work/bioinformatics/s440797/Projects/softE/MODULE_3_MATERIALS")
# 2.1 Data Configuration
DATASET = "geneturing"
# DATASET = "genehop"
DATA_PATH = f"data/{DATASET}.json"


# In[13]:


# 2.2 Model Configuration
# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API client
# NOTE: HERE we dont ever save the API key as a variable so it never gets printed
api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
api_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


# In[4]:


# 2.3 Evaluation and Logging Configuration
RESULT_FILE = f"{DATASET}_results.csv"
mlflow.set_experiment("s440797-Zimei")

# ## 3. Data Loading

# In[5]:


# 3.1 Load the JSON file
# Load the data here
try:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    print("JSON file loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file {DATA_PATH} was not found.")
    data = {}  # Initialize data as an empty dict to prevent further errors
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from the file {DATA_PATH}.")
    data = {}

# Build the TASKS variable here
TASKS = set()
if data:  # Check if data was loaded successfully
    for task_name in data.keys():
        TASKS.add(task_name)
# You can print the TASKS to verify
print("\nTasks found:")
for task in TASKS:
    print(task)


# In[ ]:


# collcet
# 3.2 Iterate through the JSON data recursively to collect each of the rows into a list
#     Each row should have a dictionary with keys of the columsn in the table above
def collect_rows(data: dict) -> list:
    rows = [
        {"task": task_name, "question": question, "answer": answer}
        for task_name, task_data in data.items()
        for question, answer in task_data.items()
    ]
    return rows


rows = collect_rows(data)
print(f"\nCollected {len(rows)} rows for task '{task_name}'.")


# 3.3 Create the pandas dataframe from the collection of rows
dataset = pd.DataFrame(rows)

# Display the dataframe
print(dataset.head())  # Display the first few rows of the dataframe


# ## 4. Model Specification

# 4.2 Draft your own system prompt for our generic genomics question answering system.
#     Replace the system message `content` below with your own.
system_content = """
You are GeneGPT, an expert genomic assistant designed to answer a wide range of questions about genes, SNPs, chromosomal locations, and related biological knowledge using reliable sources such as NCBI databases.
'You can call Eutils by: "[https://eutils.ncbi.nlm.nih.gov/entrez/eutils/{esearch|efetch|esummary}.fcgi?db={gene|snp|omim}&retmax={}&{term|id}={term|id}]".\n' \
    'esearch: input is a search term and output is database id(s).\n' \
    'efectch/esummary: input is database id(s) and output is full records or summaries that contain name, chromosome location, and other information.\n' \
    'Normally, you need to first call esearch to get the database id(s) of the search term, and then call efectch/esummary to get the information with the database id(s).\n' \
    'Database: gene is for genes, snp is for SNPs, and omim is for genetic diseases.\n\n' \
    'For DNA sequences, you can use BLAST by: "[https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi?CMD={Put|Get}&PROGRAM=blastn&MEGABLAST=on&DATABASE=nt&FORMAT_TYPE={XML|Text}&QUERY={sequence}&HITLIST_SIZE={max_hit_size}]".\n' \
    'BLAST maps a specific DNA {sequence} to its chromosome location among different specices.\n' \
## Instructions:
- Always answer using short and precise scientific terminology.
- If the answer is not available or uncertain, reply with "NA".
- Do not provide explanations unless explicitly asked.
- Only return the exact answer string, without additional commentary or punctuation.
- Respect formatting constraints provided in examples (e.g., "chr13" for chromosomes, official gene symbols like "PSMB10").
- you can also use web search to get the answer if you need.

## Context:
You are responding as part of an automated system for benchmarking LLM performance on structured genomic queries.
"""

system_message = {"role": "system", "content": system_content}
# system_message = {
#         "role": "system",
#         "content": """You are a knowledgeable assistant specialized in genomics. 
#         Your task is to provide accurate and concise answers to genomic questions using NCBI Web APIs. 
#         Please ensure your responses are clear and informative, and follow the format: 'Answer: [your answer]'. 
#         Please only provide the actual answers without any additional text or explanation. 
#         If you do it well, I will give you a reward."""
#     }


# set up the client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  # Always use it just in time
    api_version=api_version,
    azure_endpoint=api_base,
)


# ### 4.3 Setting up the few-shot examples
example_messages = [
    {"role": "user", "content": "What is the official gene symbol of LMP10?"},
    {"role": "assistant", "content": "Answer: PSMB10"},
    {
        "role": "user",
        "content": "Align the DNA sequence to the human genome:ATTCTGCCTTTAGTAATTTGATGACAGAGACTTCTTGGGAACCACAGCCAGGGAGCCACCCTTTACTCCACCAACAGGTGGCTTATATCCAATCTGAGAAAGAAAGAAAAAAAAAAAAGTATTTCTCT",
    },
    {"role": "assistant", "content": "Answer: chr15:91950805-91950932"},
    {
        "role": "user",
        "content": "Which organism does the DNA sequence come from:AGGGGCAGCAAACACCGGGACACACCCATTCGTGCACTAATCAGAAACTTTTTTTTCTCAAATAATTCAAACAATCAAAATTGGTTTTTTCGAGCAAGGTGGGAAATTTTTCGAT",
    },
    {"role": "assistant", "content": "Answer: worm"},
    {"role": "user", "content": "Convert ENSG00000215251 to official gene symbol."},
    {"role": "assistant", "content": "Answer: FASTKD5"},
    {"role": "user", "content": "Is SSBP3-AS1 a protein-coding gene?"},
    {"role": "assistant", "content": "Answer: na"},
]


# ### 4.4 Implementing the model request function

# In[16]:


# 4.4 Implementing the model request function
def query_model(
    client: AzureOpenAI,
    system_message: dict[str, str],
    few_shot_examples: List[dict[str, str]],
    user_query: str,
) -> str:
    messages = (
        [system_message] + few_shot_examples + [{"role": "user", "content": user_query}]
    )

    response = client.chat.completions.create(
        model=api_deployment,
        messages=messages,  # Adjust temperature for response variability
        max_tokens=800,  # fixed
        temperature=0.7,
        top_p=0.9
        # frequency_penalty=0.0,
        # presence_penalty=0.0
    )

    answer = response.choices[0].message.content
    return answer.split("Answer: ")[-1].strip()  # Extract everything after 'Answer :'


# ## 5. Metrics

# In[35]:


# 5.1 Implement metrics


def exact_match(pred: str, true: str) -> float:
    return 1.0 if pred == true else 0.0


def gene_disease_association(pred: list[str], true: str) -> float:
    if not true:  # Avoid division by zero
        return 0.0
    true_genes = true.split(", ")
    correct_count = sum(1 for item in pred if item in true_genes)
    return correct_count / len(true_genes) if len(true_genes) > 0 else 0.0


def disease_gene_location(pred: list[str], true: list[str]) -> float:
    if not true:  # Avoid division by zero
        return 0.0
    correct_count = sum(1 for item in pred if item in true)
    return correct_count / len(true)


def human_genome_dna_alignment(pred: str, true: str) -> float:
    if ":" not in pred:
        return 0.0
    pred_chr = pred.split(":")[0]
    pred_pos = pred.split(":")[1]
    true_chr = true.split(":")[0]
    true_pos = true.split(":")[1]

    if pred_chr != true_chr:
        return 0.0
    elif pred_pos != true_pos:
        return 0.5
    else:
        return 1.0


metric_task_map = defaultdict(
    lambda: exact_match,
    {
        "Gene disease association": gene_disease_association,
        "disease gene location": disease_gene_location,
        "Human genome DNA aligment": human_genome_dna_alignment,
    },
)


# In[23]:


# 5.2 Implement the answer mapping function
def get_answer(answer: str, task: str) -> str:
    mapper = {
        "Caenorhabditis elegans": "worm",
        "Homo sapiens": "human",
        "Danio rerio": "zebrafish",
        "Mus musculus": "mouse",
        "Saccharomyces cerevisiae": "yeast",
        "Rattus norvegicus": "rat",
        "Gallus gallus": "chicken",
    }

    if task in ["Gene location", "SNP location"]:
        if "chromosome" in answer:
            answer = "chr" + answer.split("chromosome")[-1].strip()
        if "Chromosome" in answer:
            answer = "chr" + answer.split("Chromosome")[-1].strip()
        if "chr" not in answer:
            answer = "chr" + answer
        return answer
    elif task in [
        "Gene disease association",
        "Disease gene location",
        "sequence gene alias",
    ]:
        return answer.split(", ")

    elif task == "Protein-coding genes":
        if answer == "yes" or "Yes":
            answer = "TRUE"
        elif answer=="no" or "No":
            answer = "NA"

        return answer.upper()

    elif task == "Multi-species DNA aligment":
        match = re.search(r"\((.*?)\)", answer)
        if match:
            answer = match.group(1)
        return mapper.get(answer, answer)
    else:
        return answer


# Calculate the score for a prediction against the true answer.
def get_score(pred: str, true: str, task: str) -> float:
    # Get the appropriate metric function for this task
    metric_fn = metric_task_map.get(
        task, exact_match
    )  # Default to exact match if task not found
    score = metric_fn(pred, true)
    return score


# ## 6. Evaluation Loop


# In[19]:
@dataclass
class Result:
    id: int
    task: str
    question: str
    answer: str
    raw_prediction: Optional[str]
    processed_prediction: Optional[str]
    score: Optional[float]
    success: bool


def save_results(results: List[Result], results_csv_filename: str) -> None:
    """Save the results to a CSV file."""
    df = pd.DataFrame([result.__dict__ for result in results])
    df.to_csv(results_csv_filename, index=False)


# In[36]:

print(f"Evaluating {len(dataset)} questions in the dataset...")


# 6.2 Loop over the dataset with a progress bar
# Initialize the results list


def evaluate_dataset(
    dataset: pd.DataFrame,
    system_message: str,
    client: AzureOpenAI,
    example_messages: str,
) -> tuple[List[Result], float, dict[str, float], float]:  # Updated return type
    results = []
    for index, data_row in tqdm(
        dataset.iterrows(), desc="Processing Rows", total=len(dataset)
    ):
        try:
            # Assuming `query_model` is defined in a previous cell
            question_to_ask = data_row["question"]

            # Call your model querying function
            response_from_model = query_model(
                client, system_message, example_messages, question_to_ask
            )

            # Process the answer
            processed_answer_from_model = get_answer(
                response_from_model, data_row["task"]
            )
            # Store the result
            s = get_score(
                processed_answer_from_model, data_row["answer"], data_row["task"]
            )
            results.append(
                Result(
                    id=index,
                    task=data_row["task"],
                    question=question_to_ask,
                    answer=data_row["answer"],
                    raw_prediction=response_from_model,
                    processed_prediction=processed_answer_from_model,
                    score=s,
                    success=True,
                )
            )
        # running_score = (running_score *index + s)/ (index + 1)

        except Exception as e:
            print(
                f"""Error processing index {index}
                (Question: '{data_row.get('question', 'N/A')}'): {e}"""
            )

            # Store an error result
            results.append(
                Result(
                    id=index,
                    task=data_row["task"],
                    question=question_to_ask,
                    answer=data_row.get("answer", "N/A"),
                    raw_prediction=None,
                    processed_prediction=None,
                    score=None,
                    success=False,
                )
            )
        # ## 7. Analysis

    # 7.1 Calculate the fraction of successful predictions
    successful_predictions = (
        sum(result.success for result in results) / len(results) if results else 0
    )
    print(f"Fraction of successful predictions: {successful_predictions:.2%}")


    return results, successful_predictions
    # 7.2 Calculate the overall score and the score by task
    task_scores, overall_score = score_per_task_all(results)
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


# In[46]:
# 6.3 Save the results
with mlflow.start_run(run_name="gene_turing_run"):
    # Log basic config
    mlflow.log_param("deployment_name", api_deployment)
    mlflow.log_param("version", api_version)
    mlflow.log_param("num_questions", len(dataset))
    # run the evaluation
    results, successful_predictions = evaluate_dataset(
        dataset, system_message, client, example_messages
    )
    save_results(results, RESULT_FILE)
    task_scores, overall_score = score_per_task_all(results)
    # Log the results file
    mlflow.log_artifact(RESULT_FILE)
    # Log the results
    mlflow.log_metric("successful_predictions", successful_predictions)
    mlflow.log_metric("overall_score", overall_score)
    # Log the scores by task
    for task, score in task_scores.items():
        mlflow.log_metric(f"score_{task}", score)

    # 7.3 Create a bar chart of the scores by task
    # with a horizontal line for the overall score
    plot_scores_by_task(
        task_scores, overall_score, imag_path=f"{DATASET}_scores_by_task.png"
    )
    # Log the image
    mlflow.log_artifact(f"{DATASET}_scores_by_task.png")
