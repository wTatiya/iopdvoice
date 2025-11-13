from pathlib import Path

from docx import Document

DATA_DIR = Path('data')
OUTPUT_DIR = Path('analysis/docx_text')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for path in sorted(DATA_DIR.rglob('*.docx')):
    doc = Document(path)
    text = '\n'.join(paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip())
    relative = path.relative_to(DATA_DIR)
    output_path = OUTPUT_DIR / (relative.as_posix().replace('/', '__') + '.txt')
    output_path.write_text(text)
    print(f'Wrote {output_path}')
