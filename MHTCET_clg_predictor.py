import re
import pandas as pd
from PyPDF2 import PdfReader

pdf_path = "mht-cet-cap-round-1-maharashtra-state-cutoff-2025.pdf"
rows = []

def find_rank_tuples(text):
    return re.findall(r'(\d{1,6})(?:\s*\(([\d.]+)\))?', text)

reader = PdfReader(pdf_path)

college = branch = None
categories = []

for page in reader.pages:
    text = page.extract_text() or ""
    # join broken ranks + percentiles into single line
    text = re.sub(r'(\d+)\n\(([\d.]+)\)', r'\1 (\2)', text)

    for line in text.splitlines():
        line = line.strip()

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

        # category headers (GOPENS, GSCS, …)
        cats = re.findall(r'\b[A-Z]{2,}\b', line)
        if cats:
            categories = cats
            continue

        # rank + percentile line
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
