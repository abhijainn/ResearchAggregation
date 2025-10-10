import json

# File paths — change as needed
INPUT_PATH = "arxiv-metadata-oai-snapshot.json"
OUTPUT_PATH = "nlp_arxiv_subset.json"

# (potentially) NLP-related arXiv categories
NLP_CATEGORIES = {"cs.CL", "cs.AI", "cs.LG", "stat.ML", "cs.IR", "cs.HC", "eess.AS"}

def is_nlp_paper(categories_str):
    """Returns True if any category overlaps with NLP-related ones."""
    if not categories_str:
        return False
    categories = set(categories_str.split())
    return not NLP_CATEGORIES.isdisjoint(categories)

# Streaming read + write for memory efficiency
with open(INPUT_PATH, "r", encoding="utf-8") as infile, \
     open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:

    for line in infile:
        try:
            entry = json.loads(line)
            if is_nlp_paper(entry.get("categories", "")):
                json.dump(entry, outfile, ensure_ascii=False)
                outfile.write("\n")
        except json.JSONDecodeError:
            continue  # skip malformed lines

print("Filtered NLP papers saved to:", OUTPUT_PATH)
