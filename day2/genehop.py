#!/usr/bin/env python
# coding: utf-8

# ## 1. Imports

# In[1]:


# 1.1 Place imports here
from typing import List, Optional
import re
import os
from dotenv import load_dotenv
from openai import AzureOpenAI
from collections import defaultdict
from dataclasses import dataclass
import json
import pandas as pd  # Ensure pandas is imported
from tqdm import tqdm 
import matplotlib.pyplot as plt
import mlflow
import Levenshtein
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
# ## 2. Configuration

# In[2]:

os.environ['no_proxy'] = '*'
mlflow.set_tracking_uri("http://198.215.61.34:8153/")
# os.chdir("/work/bioinformatics/s440797/Projects/softE/MODULE_3_MATERIALS")
# 2.1 Data Configuration
# DATASET = "geneturing"
DATASET = "genehop"
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
RESULT_FILE = f"{DATASET}_results.json"
mlflow.autolog()  # Enable autologging for MLflow
mlflow.set_experiment("s440797-Zimei")

# ## 3. Data Loading

# In[5]:


# 3.1 Load the JSON file
# Load the data here
try:
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    print("JSON file loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file {DATA_PATH} was not found.")
    data = {} # Initialize data as an empty dict to prevent further errors
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from the file {DATA_PATH}.")
    data = {}

# Build the TASKS variable here
TASKS = set()
if data: # Check if data was loaded successfully
    for task_name in data.keys():
        TASKS.add(task_name)
# You can print the TASKS to verify
print("\nTasks found:")
for task in TASKS:
    print(task)


#collcet
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

system_message ={
        "role": "system",
        "content": system_content
    }







# set up the client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"), # Always use it just in time
    api_version=api_version,
    azure_endpoint=api_base
)
# Load HuggingFace model and tokenizer for sentence similarity metric
# These will be used by the snp_gene_function_similarity metric
print("Loading HuggingFace BioLORD model and tokenizer...")
hf_tokenizer_name = "FremyCompany/BioLORD-2023"
hf_model_name = "FremyCompany/BioLORD-2023"
hf_tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)
hf_model = AutoModel.from_pretrained(hf_model_name)
print("BioLORD model and tokenizer loaded successfully.")

# ### 4.3 Setting up the few-shot examples
# 4.3 Appending the few-shot examples to the `messages` list
example_messages = [
    {
        "role": "user",
        "content": "What are the aliases of the gene that contains this sequnece:GTAGATGGAACTGGTAGTCAGCTGGAGAGCAGCATGGAGGCGTCCTGGGGGAGCTTCAACGCTGAGCGGGGCTGGTATGTCTCTGTCCAGCAGCCTGAAGAAGCGGAGGCCGA. Let's decompose the question to sub-questions and solve them step by step."
    },
    {
        "role": "assistant",
        "content": "Answer: SLC38A6,NAT-1,SNAT6"
    },
    {
        "role": "user",
        "content":"What is the function of the gene associated with SNP rs1217074595? Let's decompose the question to sub-questions and solve them step by step.",
    },
    {
        "role": "assistant",
        "content": "Answer: ncRNA"
    },
    {
        "role": "user",
        "content": "List chromosome locations of the genes related to Hemolytic anemia due to phosphofructokinase deficiency. Let's decompose the question to sub-questions and solve them step by step."
    },
    {
        "role": "assistant",
        "content": "Answer: 21q22.3"
    },
    {   "role": "user",
        "content":"What is the function of the gene associated with SNP rs1385096481? Let's decompose the question to sub-questions and solve them step by step." 
    },
    {
        "role": "assistant",
        "content": "Answer: This gene encodes a protein that belongs to the leucine-rich repeat and fibronectin type III domain-containing family of proteins. A similar protein in mouse, a glycosylated transmembrane protein, is thought to function in presynaptic differentiation."
    }
]


# ### 4.4 Implementing the model request function

# In[16]:


# 4.4 Implementing the model request function
def query_model(
    client: AzureOpenAI, 
    system_message: dict, 
    few_shot_examples: List[dict], 
    user_query: str
) -> str:
    messages = [system_message] + few_shot_examples + [{"role": "user", "content": user_query}]

    response = client.chat.completions.create(
        model=api_deployment,
        messages=messages,  # Adjust temperature for response variability
        max_tokens=800,  # fixed
        temperature=0.7,
        top_p=0.9
    )

    answer = response.choices[0].message.content
    return answer.split("Answer: ")[-1].strip()  # Extract everything after 'Answer :'


# ## 5. Metrics

# In[35]:


# 5.1 Implement metrics
def mean_pooling(model_output, attention_mask):
    """Pools the output of a transformer model to get a single sentence embedding."""
    token_embeddings = model_output[0]  # First element contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

def get_embeddings(sentences: list[str]):
    """Generates normalized embeddings for a list of sentences."""
    # Tokenize sentences
    encoded_input = hf_tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = hf_model(**encoded_input)
        
    # Perform pooling and normalization
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    return F.normalize(sentence_embeddings, p=2, dim=1)
def calculate_single_pair_similarity(pred: str, true: str) -> float:
    """Calculates cosine similarity for a single pair of strings."""
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0

    # Get embeddings for the prediction and ground truth
    embeddings = get_embeddings([pred, true])
    
    # Compute and return the cosine similarity
    cosine_sim = F.cosine_similarity(embeddings[0], embeddings[1], dim=0)
    return cosine_sim.item()

def calculate_list_based_similarity(pred: list[str], true: list[str]) -> float:
    """
    For each item in `pred`, finds the max cosine similarity against all items in `true`,
    then averages the results.
    """
    if not true or not pred:
        return 0.0

    # Get embeddings for all predictions and true values in a single batch
    all_texts = pred + true
    all_embeddings = get_embeddings(all_texts)
    
    pred_embeddings = all_embeddings[:len(pred)]
    true_embeddings = all_embeddings[len(pred):]

    max_similarities = []
    for pred_emb in pred_embeddings:
        # Compare one prediction embedding against all true embeddings
        similarities = F.cosine_similarity(pred_emb.unsqueeze(0), true_embeddings)
        max_similarities.append(torch.max(similarities).item())
        
    return sum(max_similarities) / len(pred)

# --- New Metric Functions ---

def gene_disease_association_similarity(pred: list[str], true: str) -> float:
    """Similarity metric for gene-disease association."""
    true_genes = true.split(", ") if true else []
    return calculate_list_based_similarity(pred, true_genes)

# Alias the generic functions for specific tasks with matching signatures
disease_gene_location_similarity = calculate_list_based_similarity
sequence_gene_alias_similarity = calculate_list_based_similarity
human_genome_dna_alignment_similarity = calculate_single_pair_similarity
snp_gene_function_similarity = calculate_single_pair_similarity

metric_task_map = defaultdict(
    lambda: calculate_single_pair_similarity, # A sensible default
    {
        "Gene disease association": gene_disease_association_similarity,
        "Disease gene location": disease_gene_location_similarity,
        "Human genome DNA aligment": human_genome_dna_alignment_similarity,
        "sequence gene alias": sequence_gene_alias_similarity,
        "SNP gene function": snp_gene_function_similarity,
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
        if answer.lower() == "yes":
            answer = "TRUE"
        elif answer.lower()=="no":
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
    metric_fn = metric_task_map[task]
     # Default to exact match if task not found
    score = metric_fn(pred, true)
    return score


# ## 6. Evaluation Loop
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

def save_results(results: List[Result], results_filename: str) -> None:
    """Save the results to a json file."""
    df = pd.DataFrame([result.__dict__ for result in results])
    df.to_json(results_filename, orient="records", lines=True, force_ascii=False)




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
# results, successful_predictions = evaluate_dataset(
#     dataset, system_message, client, example_messages
# )
# save_results(results, RESULT_FILE)
# task_scores, overall_score = score_per_task_all(results)

with mlflow.start_run(run_name="gene_hop_run"):
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



