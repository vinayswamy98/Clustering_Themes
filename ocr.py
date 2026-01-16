import easyocr
import fitz  # PyMuPDF
from PIL import Image
import numpy as np
from pathlib import Path
import io

class PDFOCRReader:
“”“Extract text from PDFs using EasyOCR and PyMuPDF”””

```
def __init__(self, languages=['en'], gpu=False):
    """
    Initialize EasyOCR reader
    languages: list of language codes, e.g., ['en', 'es', 'fr']
    gpu: True to use GPU acceleration (if available)
    """
    print("Initializing EasyOCR (this may take a moment)...")
    self.reader = easyocr.Reader(languages, gpu=gpu)
    print("EasyOCR ready!")

def pdf_to_images(self, pdf_path, dpi=300):
    """Convert PDF pages to images using PyMuPDF"""
    print(f"Converting PDF to images (DPI: {dpi})...")
    doc = fitz.open(pdf_path)
    images = []
    
    # Calculate zoom factor for desired DPI (72 is default)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    
    doc.close()
    print(f"Converted {len(images)} pages")
    return images

def ocr_image(self, image, detail=True):
    """
    Perform OCR on a single image
    detail: If True, returns coordinates; If False, returns text only
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    results = self.reader.readtext(image, detail=detail)
    return results

def extract_text_from_pdf(self, pdf_path, dpi=300, detail=False):
    """
    Extract all text from PDF
    Returns list of page texts
    """
    images = self.pdf_to_images(pdf_path, dpi)
    all_pages_text = []
    
    for i, image in enumerate(images, start=1):
        print(f"Processing page {i}/{len(images)}...")
        
        results = self.ocr_image(image, detail=detail)
        
        if detail:
            # Results include coordinates: [[bbox, text, confidence], ...]
            page_text = "\n".join([text for (bbox, text, conf) in results])
        else:
            # Results are just text strings
            page_text = "\n".join(results)
        
        all_pages_text.append({
            'page': i,
            'text': page_text
        })
    
    return all_pages_text

def extract_with_coordinates(self, pdf_path, dpi=300):
    """
    Extract text with bounding box coordinates
    Useful for understanding layout
    """
    images = self.pdf_to_images(pdf_path, dpi)
    all_results = []
    
    for i, image in enumerate(images, start=1):
        print(f"Processing page {i}/{len(images)}...")
        
        results = self.ocr_image(image, detail=True)
        
        page_data = {
            'page': i,
            'text_blocks': []
        }
        
        for bbox, text, confidence in results:
            page_data['text_blocks'].append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox,  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                'top_left': bbox[0],
                'bottom_right': bbox[2]
            })
        
        all_results.append(page_data)
    
    return all_results

def save_to_text(self, pdf_path, output_path=None, dpi=300):
    """Extract and save to text file"""
    pages = self.extract_text_from_pdf(pdf_path, dpi)
    
    if output_path is None:
        output_path = Path(pdf_path).stem + "_ocr.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for page in pages:
            f.write(f"{'='*60}\n")
            f.write(f"PAGE {page['page']}\n")
            f.write(f"{'='*60}\n\n")
            f.write(page['text'])
            f.write("\n\n")
    
    print(f"Saved to: {output_path}")
    return output_path

def extract_specific_page(self, pdf_path, page_num, dpi=300):
    """Extract text from a specific page only"""
    doc = fitz.open(pdf_path)
    
    if page_num > len(doc) or page_num < 1:
        raise ValueError(f"Page {page_num} not found. PDF has {len(doc)} pages.")
    
    # Get specific page
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    page = doc[page_num - 1]
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    doc.close()
    
    print(f"Processing page {page_num}...")
    results = self.ocr_image(img, detail=False)
    return "\n".join(results)

def extract_page_range(self, pdf_path, start_page, end_page, dpi=300):
    """Extract text from a range of pages"""
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    if start_page < 1 or end_page > total_pages:
        raise ValueError(f"Invalid page range. PDF has {total_pages} pages.")
    
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    
    results = []
    for page_num in range(start_page - 1, end_page):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        print(f"Processing page {page_num + 1}...")
        text = self.ocr_image(img, detail=False)
        results.append({
            'page': page_num + 1,
            'text': "\n".join(text)
        })
    
    doc.close()
    return results
```

def simple_example(pdf_path):
“”“Quick text extraction”””
reader = PDFOCRReader(languages=[‘en’])
pages = reader.extract_text_from_pdf(pdf_path, dpi=200)  # Lower DPI for speed

```
for page in pages:
    print(f"\n--- Page {page['page']} ---")
    print(page['text'][:500])  # Print first 500 chars
```

def save_to_file_example(pdf_path):
“”“Extract and save”””
reader = PDFOCRReader(languages=[‘en’])
output = reader.save_to_text(pdf_path, dpi=300)
print(f”Complete! Check: {output}”)

def detailed_extraction(pdf_path):
“”“Get text with coordinates and confidence”””
reader = PDFOCRReader(languages=[‘en’])
results = reader.extract_with_coordinates(pdf_path, dpi=200)

```
for page in results[:1]:  # Just first page for demo
    print(f"\nPage {page['page']}:")
    for block in page['text_blocks'][:5]:  # First 5 blocks
        print(f"  Text: {block['text']}")
        print(f"  Confidence: {block['confidence']:.2f}")
        print(f"  Position: {block['top_left']} to {block['bottom_right']}")
```

def single_page_example(pdf_path):
“”“Extract just one page”””
reader = PDFOCRReader(languages=[‘en’])
text = reader.extract_specific_page(pdf_path, page_num=1, dpi=300)
print(text)

def multilingual_example(pdf_path):
“”“OCR with multiple languages”””
# Supports: en, es, fr, de, pt, ru, ar, zh, ja, ko, etc.
reader = PDFOCRReader(languages=[‘en’, ‘es’], gpu=False)
pages = reader.extract_text_from_pdf(pdf_path)
return pages

if **name** == “**main**”:
pdf_file = “your_document.pdf”

```
# Choose your approach:

# 1. Simple extraction
# simple_example(pdf_file)

# 2. Save to file
# save_to_file_example(pdf_file)

# 3. Detailed with coordinates
# detailed_extraction(pdf_file)

# 4. Single page only
# single_page_example(pdf_file)

# 5. Multiple languages
# multilingual_example(pdf_file)

print("\nDone!")
    ```