import glob
import os
import re

md_files = glob.glob('smartfarm-paper/drafts/era-smartfarm-rag/paper/*.md')
headings = []
for f in md_files:
    if 'references.md' in f or 'deep_research_report.md' in f or 'UPDATE_PROMPT.md' in f: continue
    with open(f, 'r', encoding='utf-8') as file:
        content = file.readlines()
        for i, line in enumerate(content):
            if line.startswith('#'):
                headings.append((os.path.basename(f), i+1, line.strip()))

headings.sort(key=lambda x: (x[0], x[1]))
for h in headings:
    print(f'{h[0]} - Line {h[1]}: {h[2]}')
