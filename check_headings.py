import re
import glob

files = glob.glob('smartfarm-paper/drafts/era-smartfarm-rag/paper/*.md')
files = [f for f in files if 'references.md' not in f and 'deep_research_report' not in f and 'UPDATE_PROMPT' not in f]

for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.readlines()
        
        # Check for heading order and numbering
        for i, line in enumerate(content):
            if line.startswith('#'):
                print(f"{f}: {line.strip()}")
