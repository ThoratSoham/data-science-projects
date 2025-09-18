import pdfplumber, re, pandas as pd

pdf_path = "mht-cet-cap-round-1-maharashtra-state-cutoff-2025.pdf"
rows = []

def find_rank_tuples(text):
    # returns list of (rank_str, percentile_str_or_empty)
    return re.findall(r'(\d{1,6})(?:\s*\(([\d.]+)\))?', text)

with pdfplumber.open(pdf_path) as pdf:
    college = branch = None
    categories = []
    for page in pdf.pages:
        text = page.extract_text() or ""
        for line in text.splitlines():
            # college line: "01002 - Government College of Engineering, Amravati"
            m = re.match(r'^(\d{5})\s*-\s*(.+)$', line)
            if m:
                college = m.group(2).strip()
                continue

            # branch line: "0100219110 - Civil Engineering"
            m = re.match(r'^(\d{10})\s*-\s*(.+)$', line)
            if m:
                branch = m.group(2).strip()
                continue

            # category header line: capture all-caps tokens (GOPENS, GSCS, ...)
            cats = re.findall(r'\b[A-Z]{2,}\b', line)
            if cats:
                categories = cats
                continue

            # rank line: find numeric tokens and map to last category header
            ranks = find_rank_tuples(line)
            if ranks and categories:
                for (rank_str, pct_str), cat in zip(ranks, categories):
                    rows.append({
                        "College": college,
                        "Branch": branch,
                        "Category": cat,
                        "Rank": int(rank_str),
                        "Percentile": float(pct_str) if pct_str else None
                    })

df = pd.DataFrame(rows)

