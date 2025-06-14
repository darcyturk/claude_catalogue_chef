#!/usr/bin/env python3
"""
PDF Parser Module - The "Raw Ingredients" Processor
Extracts text, images, and metadata from PDF files with intelligent OCR integration.
"""

import fitz  # PyMuPDF
import json
import os
import sys
import argparse
from pathlib import Path
import logging
from PIL import Image
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFParser:
    def __init__(self, session_id):
        self.session_id = session_id
        self.output_dir = Path(f"data/processed_pages/{session_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = Path(f"data/images/products/{session_id}")
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_page_content(self, page, page_num):
        """Extract all content from a single PDF page"""
        logger.info(f"Processing page {page_num}")
        
        # Get page dimensions
        page_rect = page.rect
        page_width, page_height = page_rect.width, page_rect.height
        
        # Extract text with bounding boxes
        text_data = self._extract_text_with_boxes(page)
        
        # Extract images
        image_data = self._extract_images(page, page_num)
        
        # Generate page image for visual review
        page_image_path = self._generate_page_image(page, page_num)
        
        # Check if OCR is needed (sparse text detection)
        if self._needs_ocr(text_data, page_rect):
            logger.info(f"Page {page_num} appears to be image-based, applying OCR...")
            ocr_text = self._apply_ocr(page)
            text_data.extend(ocr_text)
        
        # Compile page data
        page_data = {
            "page_number": page_num,
            "page_dimensions": {
                "width": page_width,
                "height": page_height
            },
            "text_tokens": text_data,
            "images": image_data,
            "page_image_path": str(page_image_path),
            "requires_ocr": self._needs_ocr(text_data, page_rect)
        }
        
        return page_data
    
    def _extract_text_with_boxes(self, page):
        """Extract text tokens with precise bounding boxes"""
        text_tokens = []
        
        # Get words with bounding boxes
        words = page.get_text("words")
        
        for word_data in words:
            x0, y0, x1, y1, word, block_no, line_no, word_no = word_data
            
            # Get font information
            font_info = self._get_font_info(page, x0, y0, x1, y1)
            
            token = {
                "text": word,
                "bbox": [x0, y0, x1, y1],
                "font_size": font_info.get("size", 12),
                "is_bold": font_info.get("bold", False),
                "is_italic": font_info.get("italic", False),
                "block_no": block_no,
                "line_no": line_no,
                "word_no": word_no
            }
            text_tokens.append(token)
        
        return text_tokens
    
    def _get_font_info(self, page, x0, y0, x1, y1):
        """Extract font information from text spans"""
        # Sample point in the middle of the text
        sample_point = fitz.Point((x0 + x1) / 2, (y0 + y1) / 2)
        
        # Get text information at this point
        blocks = page.get_text("dict")
        
        font_info = {"size": 12, "bold": False, "italic": False}
        
        for block in blocks.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        span_rect = fitz.Rect(span["bbox"])
                        if span_rect.contains(sample_point):
                            font_info["size"] = span.get("size", 12)
                            font_flags = span.get("flags", 0)
                            font_info["bold"] = bool(font_flags & 2**4)
                            font_info["italic"] = bool(font_flags & 2**1)
                            return font_info
        
        return font_info
    
    def _extract_images(self, page, page_num):
        """Extract all images from the page"""
        images_data = []
        image_list = page.get_images()
        
        for img_index, img in enumerate(image_list):
            try:
                # Get image data
                xref, smask, width, height, bpp, colorspace, alt, name, filter_name, ext = img[:10]
                
                # Extract image
                image_doc = page.parent
                image_data = image_doc.extract_image(xref)
                image_bytes = image_data["image"]
                
                # Save image
                image_filename = f"page_{page_num:03d}_img_{img_index:03d}.{image_data['ext']}"
                image_path = self.images_dir / image_filename
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Get image bounding box
                img_bbox = self._get_image_bbox(page, xref)
                
                image_info = {
                    "id": f"img_{page_num}_{img_index}",
                    "bbox": img_bbox,
                    "path": str(image_path),
                    "metadata": {
                        "width": width,
                        "height": height,
                        "colorspace": colorspace,
                        "ext": image_data['ext'],
                        "size": len(image_bytes)
                    }
                }
                images_data.append(image_info)
                
            except Exception as e:
                logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
        
        return images_data
    
    def _get_image_bbox(self, page, xref):
        """Get bounding box for an image by its xref"""
        # This is a simplified approach - in practice, you'd need to
        # analyze the page's content stream to get exact positioning
        blocks = page.get_text("dict")
        
        for block in blocks.get("blocks", []):
            if block.get("type") == 1:  # Image block
                return block.get("bbox", [0, 0, 100, 100])
        
        # Fallback: return page dimensions
        return [0, 0, page.rect.width, page.rect.height]
    
    def _needs_ocr(self, text_tokens, page_rect):
        """Determine if OCR is needed based on text density"""
        if not text_tokens:
            return True
        
        # Calculate text coverage
        total_text_area = 0
        for token in text_tokens:
            bbox = token["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            total_text_area += area
        
        page_area = page_rect.width * page_rect.height
        text_coverage = total_text_area / page_area
        
        # If text covers less than 5% of page, likely needs OCR
        return text_coverage < 0.05
    
    def _apply_ocr(self, page):
        """Apply OCR to page using PyMuPDF's Tesseract integration"""
        ocr_tokens = []
        
        try:
            # Get OCR text with bounding boxes
            ocr_data = page.get_textpage_ocr(language="eng", dpi=300)
            words = ocr_data.extractWORDS()
            
            for word_data in words:
                x0, y0, x1, y1, word, block_no, line_no, word_no = word_data
                
                token = {
                    "text": word,
                    "bbox": [x0, y0, x1, y1],
                    "font_size": 12,  # OCR doesn't provide font info
                    "is_bold": False,
                    "is_italic": False,
                    "block_no": block_no,
                    "line_no": line_no,
                    "word_no": word_no,
                    "source": "ocr"
                }
                ocr_tokens.append(token)
                
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
        
        return ocr_tokens
    
    def _generate_page_image(self, page, page_num):
        """Generate high-resolution image of the page for visual review"""
        # Render page at high DPI for better quality
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        
        # Save as PNG
        image_path = self.output_dir / f"page_{page_num:03d}.png"
        pix.save(str(image_path))
        
        return image_path
    
    def process_pdf(self, pdf_path):
        """Process entire PDF file"""
        logger.info(f"Starting PDF processing: {pdf_path}")
        
        # Open PDF
        if pdf_path.startswith(('http://', 'https://')):
            # Download PDF from URL
            pdf_path = self._download_pdf(pdf_path)
        
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        logger.info(f"Processing {total_pages} pages")
        
        # Update state
        self._update_state("total_pages", total_pages)
        
        all_pages_data = []
        
        for page_num in range(total_pages):
            page = doc[page_num]
            page_data = self.extract_page_content(page, page_num + 1)
            
            # Save individual page data
            page_file = self.output_dir / f"page_{page_num + 1:03d}.json"
            with open(page_file, 'w') as f:
                json.dump(page_data, f, indent=2)
            
            all_pages_data.append(page_data)
            
            # Update progress
            self._update_state("processed_pages", page_num + 1)
            
            # Progress indicator
            progress = (page_num + 1) / total_pages * 100
            print(f"\rProcessing: [{('#' * int(progress/2)).ljust(50)}] {progress:.1f}%", end='')
        
        print()  # New line after progress
        
        # Save combined data
        combined_file = self.output_dir / "all_pages.json"
        with open(combined_file, 'w') as f:
            json.dump(all_pages_data, f, indent=2)
        
        doc.close()
        logger.info("PDF processing complete")
        
        return all_pages_data
    
    def _download_pdf(self, url):
        """Download PDF from URL"""
        import requests
        
        logger.info(f"Downloading PDF from {url}")
        
        response = requests.get(url)
        response.raise_for_status()
        
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        temp_file.write(response.content)
        temp_file.close()
        
        return temp_file.name
    
    def _update_state(self, key, value):
        """Update state.json file"""
        try:
            with open('state.json', 'r') as f:
                state = json.load(f)
            state[key] = value
            with open('state.json', 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update state: {e}")

def main():
    parser = argparse.ArgumentParser(description='PDF Parser - Extract content from PDF files')
    parser.add_argument('--input', required=True, help='PDF file path or URL')
    parser.add_argument('--session', required=True, help='Session ID')
    
    args = parser.parse_args()
    
    # Create parser instance
    pdf_parser = PDFParser(args.session)
    
    try:
        # Process PDF
        result = pdf_parser.process_pdf(args.input)
        
        print(f"✅ Successfully processed {len(result)} pages")
        print(f"📁 Output saved to: data/processed_pages/{args.session}/")
        
        return 0
        
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
