import re


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
        if answer == "yes" or answer == "Yes":
            answer = "TRUE"
        elif answer == "no" or answer == "No":
            answer = "NA"
        return answer.upper()

    elif task == "Multi-species DNA aligment":
        match = re.search(r"\((.*?)\)", answer)
        if match:
            answer = match.group(1)
        return mapper.get(answer, answer)
    else:
        return answer


def test_get_answer():
    assert get_answer("chr1", "SNP location") == "chr1"
    assert get_answer("1", "SNP location") == "chr1"
    assert get_answer("chromosome 1", "Gene location") == "chr1"
    assert get_answer("chromosome 2", "Gene location") == "chr2"
    assert get_answer("Caenorhabditis elegans", "Multi-species DNA alignment") == "worm"
    assert get_answer("Homo sapiens", "Multi-species DNA alignment") == "human"
    assert get_answer("Yes", "Protein-coding genes") == "TRUE"  
    assert get_answer("No", "Protein-coding genes") == "NA"   
    assert get_answer("gene1, gene2", "Gene disease association") == ["gene1", "gene2"]
    assert get_answer("location1, location2", "Disease gene location") == [
        "location1",
        "location2",
    ]


# collcet
def collect_rows(data: dict) -> list:
    rows = [
        {"task": task_name, "question": question, "answer": answer}
        for task_name, task_data in data.items()
        for question, answer in task_data.items()
    ]
    return rows


def test_collect_rows():
    data = {
        "Personal Information": {
            "What is your name?": "Alice",
            "What is your age?": "30",
        }
    }

    rows = collect_rows(data)

    expected_rows = [
        {
            "task": "Personal Information",
            "question": "What is your name?",
            "answer": "Alice",
        },
        {
            "task": "Personal Information",
            "question": "What is your age?",
            "answer": "30",
        },
    ]

    assert rows == expected_rows


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


def test_human_genome_dna_alignment():
    assert human_genome_dna_alignment("chr1:100", "chr1:100") == 1.0
    assert human_genome_dna_alignment("chr1:100", "chr1:200") == 0.5
    assert human_genome_dna_alignment("chr1:100", "chr2:100") == 0.0
    assert human_genome_dna_alignment("chr2:150", "chr2:150") == 1.0
    assert human_genome_dna_alignment("chr2:150", "chr1:150") == 0.0
    assert human_genome_dna_alignment("chr1:300", "chr1:300") == 1.0
