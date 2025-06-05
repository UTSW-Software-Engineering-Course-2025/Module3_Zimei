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
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import blast_Eutils
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
NCBI_API_KEY = os.getenv("NCBI_API_KEY")

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
SYSTEM_PROMPT_TOOL_CALLING = """
You are a helpful AI assistant designed to retrieve biomedical information from NCBI using specific tools.
Your goal is to answer the user's question by calling a sequence of tools if necessary.

Workflow:
1. Examine the user's question.
2. If a tool can help answer the question, decide which tool to use and what arguments to pass.
3. After all necessary tool calls are made and you have the information, provide a final, concise answer to the user.
   Only provide the direct answer, without explanations or conversational filler, unless the question implies a need for explanation.
   For list-based answers (e.g., aliases, associated SNPs/diseases), provide them as a comma-separated string.
   For boolean answers (e.g. protein-coding), respond with "true" or "na".
   For location answers, use "chrX" or "chrX:start-end" format. If not found, use "NA".

Remember:
- You can make multiple tool calls in sequence if needed.
- The response from each tool call will be provided back to you.
- Once you have sufficient information, provide the final answer directly without requesting more tool calls.
- If you determine that tools cannot answer the question or if tools return errors/no data, provide "NA" as the final answer.
"""







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

# Tool Definitions (Schema for OpenAI API)
TOOLS_DEFINITION_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_gene_uid",
            "description": "Searches NCBI Gene DB for a UID given a gene query term (e.g., official symbol, alias, Ensembl ID). Use this to get a UID before summarizing gene details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_term": {"type": "string", "description": "The gene identifier or name to search for."},
                },
                "required": ["query_term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_gene_details_by_uid",
            "description": "Retrieves a structured summary of gene information (official symbol, description, organism, aliases, type, location) for a given NCBI gene UID.",
            "parameters": {
                "type": "object",
                "properties": {"uid": {"type": "string", "description": "The NCBI gene ID (UID)."}},
                "required": ["uid"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_snp_uid",
            "description": "Searches NCBI SNP DB for a UID given an rsID. Use this to get a UID before summarizing SNP details.",
            "parameters": {
                "type": "object",
                "properties": {"rsid": {"type": "string", "description": "The rsID of the SNP (e.g., 'rs12345')."}},
                "required": ["rsid"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_snp_details_by_uid",
            "description": "Retrieves a structured summary of SNP information (chromosome, position, clinical significance, related genes) for a given NCBI SNP UID.",
            "parameters": {
                "type": "object",
                "properties": {"uid": {"type": "string", "description": "The NCBI SNP ID (UID)."}},
                "required": ["uid"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_blast_sequence",
            "description": "Submits a DNA sequence to NCBI BLASTn against the 'nt' database for alignment. Returns a Request ID (RID).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sequence": {"type": "string", "description": "The DNA sequence to align."},
                    "program": {"type": "string", "description": "BLAST program, e.g., 'blastn'. Default: 'blastn'", "default": "blastn"},
                    "database": {"type": "string", "description": "BLAST database, e.g., 'nt'. Default: 'nt'", "default": "nt"},
                    "hitlist_size": {"type": "integer", "description": "Max number of hits. Default: 1", "default": 1}
                },
                "required": ["sequence"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_blast_results",
            "description": "Retrieves BLAST results for a given Request ID (RID). Polls for readiness. Returns raw BLAST text output if ready.",
            "parameters": {
                "type": "object",
                "properties": {"rid": {"type": "string", "description": "The BLAST Request ID (RID)."}},
                "required": ["rid"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_omim_for_gene_related_diseases",
            "description": "Searches OMIM for diseases related to a gene term. Returns a list of disease names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gene_term": {"type": "string", "description": "The gene symbol or name to search for related diseases."},
                    "retmax": {"type": "integer", "description": "Maximum number of disease entries to return. Default: 3", "default": 3}
                },
                "required": ["gene_term"]
            }
        }
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
MAX_TOOL_CALL_TURNS = 8
def evaluate_dataset(
    dataset: pd.DataFrame,
    client: AzureOpenAI,
    model_deployment_name: str,
    tools_schema: List[dict],
    available_tool_functions: dict[str, callable],
    ncbi_api_key: Optional[str]
) -> tuple[List[Result], float, dict[str, float], float]: 

    results = []
    for index, data_row in tqdm(dataset.iterrows(), desc="Processing Rows", total=len(dataset)):
        question_to_ask = data_row["question"]
        task_name = data_row["task"]
        ground_truth_answer = data_row["answer"]
        
        final_content_for_processing = "NA"
        full_message_history = []
        success = False # Using 'success'
        s = 0.0         # Using 's' for score
        processed_answer_from_model = None # Using 'processed_answer_from_model'

        messages_for_llm = [ # Using 'messages_for_llm'
            {"role": "system", "content": SYSTEM_PROMPT_TOOL_CALLING},
            {"role": "user", "content": question_to_ask}
        ]
        full_message_history.extend(messages_for_llm)

        try:
            for turn in range(MAX_TOOL_CALL_TURNS):
                # print(f"Debug: Row {index}, Turn {turn+1}: Sending messages to LLM: {messages_for_llm}") # Optional print for debugging
                response = client.chat.completions.create(
                    model=model_deployment_name,
                    messages=messages_for_llm,
                    tools=tools_schema,
                    tool_choice="auto",
                    temperature=0,
                )
                response_message = response.choices[0].message
                
                assistant_msg_to_log = {"role": "assistant", "content": response_message.content}
                if response_message.tool_calls:
                    assistant_msg_to_log["tool_calls"] = [
                        {"id": tc.id, "type": tc.type, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in response_message.tool_calls
                    ]
                messages_for_llm.append(assistant_msg_to_log)
                full_message_history.append(assistant_msg_to_log)

                if response_message.tool_calls:
                    print(f"Info: Row {index}, Turn {turn+1}: LLM tool calls: {[tc.function.name for tc in response_message.tool_calls]}") # Changed to print
                    for tool_call in response_message.tool_calls:
                        function_name = tool_call.function.name
                        tool_message_content = ""
                        if function_name in available_tool_functions:
                            function_to_call = available_tool_functions[function_name]
                            try:
                                function_args_str = tool_call.function.arguments
                                function_args = json.loads(function_args_str)
                                if "api_key" in function_to_call.__code__.co_varnames:
                                    function_args["api_key"] = ncbi_api_key
                                print(f"Info: Row {index}: Calling tool '{function_name}' with args: {function_args}") # Changed to print
                                tool_message_content = function_to_call(**function_args)
                                print(f"Info: Row {index}: Tool '{function_name}' response: {str(tool_message_content)[:100]}") # Changed to print
                            except json.JSONDecodeError:
                                print(f"Error: Row {index}: Parsing args for {function_name}: {function_args_str}") # Changed to print
                                tool_message_content = json.dumps({"error": f"Invalid arguments format: {function_args_str}"})
                            except TypeError as te:
                                print(f"Error: Row {index}: TypeError calling {function_name} with {function_args}: {te}") # Changed to print
                                tool_message_content = json.dumps({"error": f"TypeError in {function_name}: {str(te)}"})
                            except Exception as e_tool:
                                print(f"Error: Row {index}: Executing tool {function_name}: {e_tool}") # Changed to print
                                tool_message_content = json.dumps({"error": f"Exception in {function_name}: {str(e_tool)}"})
                        else:
                            print(f"Error: Row {index}: LLM requested unknown function '{function_name}'") # Changed to print
                            tool_message_content = json.dumps({"error": f"Function '{function_name}' not found."})
                        
                        messages_for_llm.append({
                            "tool_call_id": tool_call.id, "role": "tool",
                            "name": function_name, "content": tool_message_content,
                        })
                        full_message_history.append(messages_for_llm[-1])
                else:
                    final_content_for_processing = response_message.content if response_message.content else "NA"
                    print(f"Info: Row {index}, Turn {turn+1}: LLM final answer: {final_content_for_processing}") # Changed to print
                    success = True
                    break 
            
            if not success:
                print(f"Warning: Row {index}: Max turns ({MAX_TOOL_CALL_TURNS}) reached. Using last assistant content or 'NA'.") # Changed to print
                final_content_for_processing = next((m["content"] for m in reversed(messages_for_llm) if m["role"] == "assistant" and m.get("content")), "NA")

            processed_answer_from_model = get_answer(final_content_for_processing, task_name)
            s = get_score(processed_answer_from_model, ground_truth_answer, task_name)
            success = True 

        except Exception as e_main:
            print(f"Error: Critical error processing index {index} (Q: '{question_to_ask}'): {e_main}") # Changed to print
            # Consider logging traceback for e_main with import traceback; traceback.print_exc()
            final_content_for_processing = f"Error: {str(e_main)}"
            success = False
            s = 0.0
            processed_answer_from_model = get_answer("NA", task_name)

        results.append(Result(
            id=index, 
            task=task_name, 
            question=question_to_ask, 
            answer=ground_truth_answer,
            raw_prediction=json.dumps(full_message_history, indent=2),
            processed_prediction=processed_answer_from_model,
            score=s, 
            success=success
        ))

    successful_predictions = (sum(r.success for r in results) / len(results) if results else 0)

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
        dataset=dataset, 
        client = client, 
        model_deployment_name=api_deployment,
        tools_schema=TOOLS_DEFINITION_SCHEMA,
        available_tool_functions=blast_Eutils.AVAILABLE_FUNCTIONS,
        ncbi_api_key=NCBI_API_KEY
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



