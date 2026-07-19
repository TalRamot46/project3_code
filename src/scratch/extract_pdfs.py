import PyPDF2
import sys
import os

# Extract bath_temperature_derivation
bath_path = r'c:\Users\TLP-001\Documents\GitHub\project3_docs\fitting\bath_temperature_derivation (1).pdf'
sub_path = r'c:\Users\TLP-001\Documents\GitHub\project3_docs\fitting\sub (2).pdf'

out_dir = r'c:\Users\TLP-001\Documents\GitHub\project3_code\scratch'

# Extract bath_temperature_derivation
reader = PyPDF2.PdfReader(bath_path)
with open(os.path.join(out_dir, 'bath_temperature_text.txt'), 'w', encoding='utf-8') as f:
    f.write(f'Total pages: {len(reader.pages)}\n')
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        f.write(f'\n=== PAGE {i+1} ===\n')
        f.write(text + '\n')

# Extract sub pages 27-31 (0-indexed: 26-30)
reader2 = PyPDF2.PdfReader(sub_path)
with open(os.path.join(out_dir, 'sub_pages_27_31_text.txt'), 'w', encoding='utf-8') as f:
    f.write(f'Total pages in sub: {len(reader2.pages)}\n')
    for i in range(26, min(31, len(reader2.pages))):
        text = reader2.pages[i].extract_text()
        f.write(f'\n=== PAGE {i+1} ===\n')
        f.write(text + '\n')

print("Done extracting PDFs")
