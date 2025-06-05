# blast_tool.py (Adapted for OpenAI Tool Calling)
import requests
import time
import random
import logging
import re
import json
from typing import Optional, Dict, List, Any

# Configure basic logging for the tool
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - blast_tool - %(message)s')

# --- NCBI API Interaction (Common) ---
EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
BLAST_BASE_URL = "https://blast.ncbi.nlm.nih.gov/blast/Blast.cgi"

def call_api(url: str, params: Optional[Dict[str, str]] = None, is_eutils: bool = False, api_key_val: Optional[str] = None, timeout: int = 60, max_retries: int = 3) -> bytes:
    """
    Perform an HTTP GET request. Includes API key for E-utils if provided.
    Retries with exponential back-off.
    """
    base_delay = 1
    if params is None:
        params = {}
    if is_eutils and api_key_val:
        params["api_key"] = api_key_val

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                logging.warning(f"API request failed (attempt {attempt+1}/{max_retries}): Retrying in {delay:.1f}s...")
                time.sleep(delay)
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            status = getattr(e.response, "status_code", None)
            if status in [429, 500, 502, 503, 504] or isinstance(e, requests.exceptions.Timeout):
                if attempt == max_retries - 1:
                    logging.error(f"API request failed after {max_retries} retries: {url} params={params} - Error: {e}")
                    raise
                continue
            else:
                logging.error(f"API request failed with non-retryable error: {url} params={params} - Error: {e}")
                raise
    raise RuntimeError(f"API request failed after {max_retries} retries: {url} params={params}")

# --- Tool Implementations ---

# BLAST Tools
def _extract_blast_rid(response_bytes: bytes) -> Optional[str]:
    if not response_bytes: return None
    match = re.search(r'RID\s*=\s*([A-Z0-9\-]+)', response_bytes.decode('utf-8', errors='ignore')) # Keep robust RID extraction
    return match.group(1).strip() if match else None

def submit_blast_sequence(sequence: str, program: str = "blastn", database: str = "nt", hitlist_size: int = 1) -> str:
    """Submits a sequence to BLAST and returns the RID. For OpenAI tool calling."""
    logging.info(f"TOOL: submit_blast_sequence called with sequence: {sequence[:30]}...")
    params = {
        "CMD": "Put", "PROGRAM": program, "DATABASE": database, "QUERY": sequence,
        "HITLIST_SIZE": str(hitlist_size), "MEGABLAST": "on", "FORMAT_TYPE": "Text"
    }
    try:
        response_content = call_api(BLAST_BASE_URL, params=params)
        rid = _extract_blast_rid(response_content)
        if rid:
            logging.info(f"TOOL: submit_blast_sequence got RID: {rid}")
            return json.dumps({"rid": rid, "status": "submitted"})
        else:
            logging.error(f"TOOL ERROR: submit_blast_sequence could not extract RID. Response: {response_content[:200]}")
            return json.dumps({"error": "Could not extract BLAST RID", "details": response_content[:200].decode('utf-8', errors='ignore')})
    except Exception as e:
        logging.error(f"TOOL ERROR: submit_blast_sequence failed: {e}")
        return json.dumps({"error": str(e)})

def get_blast_results(rid: str, initial_wait_time: int = 20, retries_for_ready: int = 3, delay_between_retries: int = 10) -> str:
    """Retrieves BLAST results for a given RID. Waits if not ready. For OpenAI tool calling."""
    logging.info(f"TOOL: get_blast_results called for RID: {rid}")
    params = {"CMD": "Get", "RID": rid, "FORMAT_TYPE": "Text"}
    
    # Simplified polling based on your earlier request for simpler blast_get_request
    current_wait = initial_wait_time
    for attempt in range(retries_for_ready + 1):
        if attempt == 0: # First attempt uses initial_wait_time
             logging.info(f"TOOL: Waiting {current_wait}s for BLAST RID {rid} (initial).")
             time.sleep(current_wait)
        elif attempt > 0: # Subsequent retries
             logging.info(f"TOOL: BLAST results not ready. Waiting an additional {delay_between_retries}s for RID {rid} (attempt {attempt}/{retries_for_ready}).")
             time.sleep(delay_between_retries)

        try:
            response_bytes = call_api(BLAST_BASE_URL, params=params, timeout=120)
            response_str = response_bytes.decode('utf-8', errors='ignore')

            if "Status=WAITING" in response_str:
                if attempt < retries_for_ready:
                    logging.info(f"TOOL: BLAST RID {rid} status is WAITING.")
                    continue # Go to next retry iteration
                else: # Max retries reached for waiting
                    logging.warning(f"TOOL: BLAST RID {rid} still WAITING after max retries.")
                    return json.dumps({"rid": rid, "status": "WAITING", "message": "Results not ready after polling."})
            elif "Status=FAILED" in response_str:
                logging.error(f"TOOL ERROR: BLAST RID {rid} search FAILED. Response: {response_str[:200]}")
                return json.dumps({"rid": rid, "status": "FAILED", "error_details": response_str[:500]})
            elif "Status=READY" in response_str:
                logging.info(f"TOOL: BLAST RID {rid} status is READY.")
                if "No hits found" in response_str:
                    return json.dumps({"rid": rid, "status": "READY", "result": "No hits found"})
                # For actual alignment parsing, we'll need a specific tool or the LLM can parse the text
                return json.dumps({"rid": rid, "status": "READY", "result_text": response_str})
            elif "Sbjct" in response_str and "Query" in response_str: # Heuristic for ready if no clear status
                logging.info(f"TOOL: BLAST RID {rid} - No explicit status, but appears to contain alignment data.")
                return json.dumps({"rid": rid, "status": "READY_HEURISTIC", "result_text": response_str})
            else: # Ambiguous or unknown status
                logging.warning(f"TOOL: BLAST RID {rid} - Ambiguous status. Assuming not ready or error. Response: {response_str[:200]}")
                if attempt < retries_for_ready: continue
                return json.dumps({"rid": rid, "status": "UNKNOWN", "error_details": response_str[:500]})
        except Exception as e:
            logging.error(f"TOOL ERROR: get_blast_results failed for RID {rid}: {e}")
            if attempt < retries_for_ready: continue
            return json.dumps({"rid": rid, "status": "ERROR", "error": str(e)})
            
    return json.dumps({"rid": rid, "status": "TIMEOUT_POLLING", "message": "Max polling attempts reached for BLAST results."})


# E-utils Tools (adapted from notebook and previous geneturing)
def _eutils_search_ids(db: str, term: str, retmax: int = 1, api_key: Optional[str] = None) -> List[str]:
    """Helper for E-utils esearch, returns list of UIDs."""
    params = {"db": db, "term": term, "retmax": str(retmax), "retmode": "json"}
    try:
        response_bytes = call_api(EUTILS_BASE_URL + "esearch.fcgi", params=params, is_eutils=True, api_key_val=api_key)
        response_json = json.loads(response_bytes.decode('utf-8'))
        return response_json.get("esearchresult", {}).get("idlist", [])
    except Exception: # Simplified error handling for helper
        return []

def _eutils_get_summary(db: str, id_list: List[str], api_key: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Helper for E-utils esummary, returns 'result' dict."""
    if not id_list: return None
    params = {"db": db, "id": ",".join(id_list), "retmode": "json"}
    try:
        response_bytes = call_api(EUTILS_BASE_URL + "esummary.fcgi", params=params, is_eutils=True, api_key_val=api_key)
        return json.loads(response_bytes.decode('utf-8')).get("result", {})
    except Exception: # Simplified error handling for helper
        return None

# Tool: Search for Gene UID
def search_gene_uid(query_term: str, api_key: Optional[str] = None) -> str:
    """Searches NCBI Gene DB for a UID given a gene query term (e.g., symbol, Ensembl ID)."""
    logging.info(f"TOOL: search_gene_uid called with query_term: {query_term}")
    uids = _eutils_search_ids(db="gene", term=query_term, retmax=1, api_key=api_key)
    if uids:
        logging.info(f"TOOL: search_gene_uid found UID: {uids[0]} for term '{query_term}'")
        return json.dumps({"uid": uids[0], "query_term": query_term})
    else:
        logging.warning(f"TOOL: search_gene_uid found no UID for '{query_term}'")
        return json.dumps({"error": f"No UID found for gene term '{query_term}'"})

# Tool: Summarize Gene Details using UID
def summarize_gene_details_by_uid(uid: str, api_key: Optional[str] = None) -> str:
    """Retrieves summary for a given NCBI gene UID."""
    logging.info(f"TOOL: summarize_gene_details_by_uid called for UID: {uid}")
    summary_result = _eutils_get_summary(db="gene", id_list=[uid], api_key=api_key)
    if summary_result and uid in summary_result:
        gene_data = summary_result[uid]
        # Extract key information as in the notebook
        extracted = {
            "uid": uid,
            "official_symbol": gene_data.get("nomenclaturesymbol"), # Official symbol from nomenclature
            "description": gene_data.get("description"),
            "organism": gene_data.get("organism", {}).get("scientificname"),
            "aliases": gene_data.get("otheraliases",""),
            "type_of_gene": gene_data.get("type_of_gene"),
            "map_location": gene_data.get("maplocation"), # Detailed location
            "chromosome": gene_data.get("chromosome"), # Simpler chromosome
            "summary_text": gene_data.get("summary")
        }
        logging.info(f"TOOL: summarize_gene_details_by_uid successful for UID: {uid}")
        return json.dumps(extracted)
    else:
        logging.warning(f"TOOL: summarize_gene_details_by_uid found no summary for UID: {uid}")
        return json.dumps({"error": f"No summary found for gene UID {uid}"})

# Tool: Search for SNP UID
def search_snp_uid(rsid: str, api_key: Optional[str] = None) -> str:
    """Searches NCBI SNP DB for a UID given an rsID."""
    logging.info(f"TOOL: search_snp_uid called with rsid: {rsid}")
    if not rsid or not rsid.lower().startswith("rs"):
        return json.dumps({"error": f"Invalid rsID format: {rsid}"})
    uids = _eutils_search_ids(db="snp", term=rsid, retmax=1, api_key=api_key)
    if uids:
        logging.info(f"TOOL: search_snp_uid found UID: {uids[0]} for rsID '{rsid}'")
        return json.dumps({"uid": uids[0], "rsid": rsid})
    else:
        logging.warning(f"TOOL: search_snp_uid found no UID for '{rsid}'")
        return json.dumps({"error": f"No UID found for SNP rsID '{rsid}'"})

# Tool: Summarize SNP Details using UID
def summarize_snp_details_by_uid(uid: str, api_key: Optional[str] = None) -> str:
    """Retrieves summary for a given NCBI SNP UID."""
    logging.info(f"TOOL: summarize_snp_details_by_uid called for UID: {uid}")
    summary_result = _eutils_get_summary(db="snp", id_list=[uid], api_key=api_key)
    if summary_result and uid in summary_result:
        snp_data = summary_result[uid]
        extracted = {
            "uid": uid,
            "rsid_from_summary": f"rs{snp_data.get('refsnp_id')}" if snp_data.get('refsnp_id') else snp_data.get('caption'),
            "chromosome": snp_data.get("chr"),
            "chr_position": snp_data.get("chrpos"),
            "organism": snp_data.get("organism_common"), # e.g. human
            "clinical_significance": snp_data.get("clinical_significance"),
            "gene_summary": snp_data.get("genes", []) # List of gene dicts associated
        }
        logging.info(f"TOOL: summarize_snp_details_by_uid successful for UID: {uid}")
        return json.dumps(extracted)
    else:
        logging.warning(f"TOOL: summarize_snp_details_by_uid found no summary for UID: {uid}")
        return json.dumps({"error": f"No summary found for SNP UID {uid}"})


# Tool: Search for Diseases related to a gene term (using OMIM)
def search_omim_for_gene_related_diseases(gene_term: str, retmax: int = 3, api_key: Optional[str] = None) -> str:
    """Searches OMIM for diseases related to a gene term."""
    logging.info(f"TOOL: search_omim_for_gene_related_diseases for gene: {gene_term}")
    # Search OMIM with the gene term. This might find the gene's OMIM entry or related phenotype entries.
    # A more direct approach would be to first get gene UID, then use elink to find OMIM links,
    # but keeping it simpler with a direct OMIM search for now.
    uids = _eutils_search_ids(db="omim", term=f"{gene_term}[Gene Symbol] AND \"phenotype\"[Properties]", retmax=retmax, api_key=api_key)
    if not uids:
        # Fallback: search without "phenotype" property if initial search yields nothing
        uids = _eutils_search_ids(db="omim", term=gene_term, retmax=retmax, api_key=api_key)
        if not uids:
            logging.warning(f"TOOL: No OMIM UIDs found for gene term '{gene_term}'")
            return json.dumps({"error": f"No OMIM entries found for gene term '{gene_term}'"})

    summaries = _eutils_get_summary(db="omim", id_list=uids, api_key=api_key)
    if not summaries:
        return json.dumps({"error": f"Could not retrieve OMIM summaries for UIDs of '{gene_term}'"})

    disease_info = []
    for uid in uids:
        if uid in summaries:
            entry = summaries[uid]
            title = entry.get("title", "")
            # Clean title (heuristic from before)
            cleaned_title = re.sub(r"^\s*\[?\s*#?\s*\d+\s*\]?\s*", "", title).strip()
            cleaned_title = re.sub(r";.*$", "", cleaned_title).strip()
            if cleaned_title:
                disease_info.append({"omim_id": uid, "disease_name": cleaned_title})
    
    if disease_info:
        return json.dumps({"gene_term": gene_term, "related_diseases": disease_info})
    else:
        return json.dumps({"error": f"No clear disease names found in OMIM summaries for '{gene_term}'"})

# Mapping of tool names (as LLM will call them) to functions
AVAILABLE_FUNCTIONS = {
    "search_gene_uid": search_gene_uid,
    "summarize_gene_details_by_uid": summarize_gene_details_by_uid,
    "search_snp_uid": search_snp_uid,
    "summarize_snp_details_by_uid": summarize_snp_details_by_uid,
    "submit_blast_sequence": submit_blast_sequence,
    "get_blast_results": get_blast_results,
    "search_omim_for_gene_related_diseases": search_omim_for_gene_related_diseases,
    # Add more mappings here as new tool functions are created
}
