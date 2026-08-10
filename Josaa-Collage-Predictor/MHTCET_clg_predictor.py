pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
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
    text = re.sub(r'(\d+)\n\(([\d.]+)\)', r'\1 (\2)', text)

    for line in text.splitlines():
        line = line.strip()

        m = re.match(r'^(\d{5})\s*-\s*(.+)$', line)
        if m:
            college = m.group(2).strip()
            continue

        m = re.match(r'^(\d{10})\s*-\s*(.+)$', line)
        if m:
            branch = m.group(2).strip()
            continue

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


def catinfo():
  print("""['CAP', 'GOPENS', 'GSCS', 'GSTS', 'GVJS', 'GOBCS', 'GSEBCS',
       'LOPENS', 'LSCS', 'LSTS', 'LVJS', 'LOBCS', 'LSEBCS', 'PWDOPENS',
       'PWDOBCS', 'DEFOPENS', 'DEFOBCS', 'TFWS', 'PWDRSCS', 'DEFROBCS',
       'ORPHAN', 'EWS', 'SDEFROBCS', 'SORPHAN', 'SEWS', 'GOPENH', 'GSCH',
       'GOBCH', 'LOPENH', 'LOBCH', 'GSTH', 'GOPENO', 'GSCO', 'GVJO',
       'GSEBCO', 'LOPENO', 'GVJH', 'GSEBCH', 'GOBCO', 'LSCO', 'LSCH',
       'LSTH', 'LVJH', 'LSEBCH', 'LSTO', 'PWDOPENH', 'PWDOBCH', 'GSTO',
       'LVJO', 'LOBCO', 'LSEBCO', 'DEFSCS', 'DEFRSCS', 'II', 'MI',
       'PWDROBC', 'DEFRVJS', 'DEFSEBCS', 'SDEFRVJS', 'PWDRSTS', 'PWDSCS',
       'PWDRSCH', 'DEFRSTS', 'PWDSEBCS', 'PWDSTS', 'DEFSTS']""")
per = input("Enter percentile: ")
cat = input("Enter category \n (enter cat for viewing all the categories): ")
if cat == "cat":
  catinfo()
  cat = input("Enter category: ")
rank_category_condition = (pd.to_numeric(df["Percentile"], errors="coerce") <= float(per)) & \
                          (df["Category"].str.upper() == cat)
eligible = df[rank_category_condition]

c = eligible[["College", "Branch", "Rank", "Category", "Percentile"]]
c = c.sort_values(by="Percentile", ascending=False)
c
