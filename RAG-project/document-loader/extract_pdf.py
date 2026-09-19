import os
import sys
from pypdf import PdfReader, PdfWriter

# Fix Unicode encoding for Windows terminal
sys.stdout.reconfigure(encoding='utf-8')

# Define paths
current_dir = os.path.dirname(os.path.abspath(__file__))
input_pdf = os.path.join(current_dir, "deep-learning.pdf")
output_pdf = os.path.join(current_dir, "Deep_Learning_Pages_16_to_76.pdf")

# Verify input file exists
if not os.path.exists(input_pdf):
    print(f"Error: Input file '{input_pdf}' not found.")
    sys.exit(1)

print(f"Reading: {input_pdf}")
reader = PdfReader(input_pdf)
total_pages = len(reader.pages)
print(f"Total pages in source PDF: {total_pages}")

writer = PdfWriter()

# Book printed page vs PDF sheet mapping:
# In deep-learning.pdf, Book Page 1 starts at PDF Index 16 (PDF Sheet 17).
# Therefore, offset is +15:
# Book Page 16 -> PDF Index 31 (Sheet 32)
# Book Page 17 (Chapter 2 start) -> PDF Index 32 (Sheet 33)
# Book Page 76 (Chapter 4 end)   -> PDF Index 91 (Sheet 92)

book_page_start = 16
book_page_end = 76

start_index = book_page_start + 15  # Index 31 (Book page 16)
end_index = book_page_end + 15      # Index 91 (Book page 76)

print(f"Extracting Book Pages {book_page_start} to {book_page_end}...")
print(f"Source PDF indices: {start_index} to {end_index} (inclusive)")

for i in range(start_index, min(end_index + 1, total_pages)):
    writer.add_page(reader.pages[i])

with open(output_pdf, "wb") as f:
    writer.write(f)

print(f"\nPDF Saved to: {output_pdf}")
print(f"Total pages in output PDF: {len(writer.pages)}")
