#!/usr/bin/env python3
"""
Barcode Scanner Module - The "Product ID Detective"
Detects and decodes barcodes from product images and page regions.
"""

import cv2
import numpy as np
from pyzbar import pyzbar
import json
import os
import sys
import argparse
from pathlib import Path
import logging
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Tuple, Optional
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BarcodeScanner:
    def __init__(self, session_id):
        self.session_id = session_id
        self.input_dir = Path(f"data/processed_pages/{session_id}")
        self.images_dir = Path(f"data/images/products/{session_id}")
        self.output_dir = Path(f"data/processed_pages/{session_id}")
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Barcode detection parameters
        self.barcode_types = [
            pyzbar.ZBarSymbol.EAN13,
            pyzbar.ZBarSymbol.EAN8,
            pyzbar.ZBarSymbol.UPCA,
            pyzbar.ZBarSymbol.UPCE,
            pyzbar.ZBarSymbol.CODE128,
            pyzbar.ZBarSymbol.CODE39,
            pyzbar.ZBarSymbol.CODE93,
            pyzbar.ZBarSymbol.CODABAR,
            pyzbar.ZBarSymbol.ITF25,
            pyzbar.ZBarSymbol.DATAMATRIX,
            pyzbar.ZBarSymbol.QRCODE
        ]
        
        # Image preprocessing parameters
        self.preprocessing_methods = [
            "original",
            "grayscale",
            "binary_threshold",
            "adaptive_threshold",
            "gaussian_blur",
            "sharpen",
            "contrast_enhance",
            "morphological_ops"
        ]
    
    def detect_barcodes_in_image(self, image_path: str, region_bbox: Optional[List[float]] = None) -> List[Dict]:
        """Detect barcodes in an image, optionally within a specific region"""
        logger.info(f"Scanning for barcodes in {Path(image_path).name}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.warning(f"Could not load image: {image_path}")
            return []
        
        original_image = image.copy()
        
        # Crop to region if specified
        if region_bbox:
            x0, y0, x1, y1 = map(int, region_bbox)
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            image = image[y0:y1, x0:x1]
            offset = (x0, y0)
        else:
            offset = (0, 0)
        
        all_barcodes = []
        
        # Try different preprocessing methods
        for method in self.preprocessing_methods:
            processed_image = self._preprocess_image(image, method)
            barcodes = self._scan_image_for_barcodes(processed_image, method, offset)
            
            if barcodes:
                all_barcodes.extend(barcodes)
                logger.info(f"Found {len(barcodes)} barcode(s) with method: {method}")
        
        # Remove duplicates based on data and position
        unique_barcodes = self._deduplicate_barcodes(all_barcodes)
        
        # Validate barcodes
        validated_barcodes = self._validate_barcodes(unique_barcodes)
        
        # Save detection visualization
        if validated_barcodes:
            self._save_detection_visualization(
                original_image, validated_barcodes, image_path, region_bbox
            )
        
        return validated_barcodes
    
    def _preprocess_image(self, image: np.ndarray, method: str) -> np.ndarray:
        """Apply preprocessing method to improve barcode detection"""
        if method == "original":
            return image
        
        elif method == "grayscale":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        elif method == "binary_threshold":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return binary
        
        elif method == "adaptive_threshold":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            adaptive = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            return adaptive
        
        elif method == "gaussian_blur":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            return blurred
        
        elif method == "sharpen":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(gray, -1, kernel)
            return sharpened
        
        elif method == "contrast_enhance":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
            return enhanced
        
        elif method == "morphological_ops":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            return processed
        
        return image
    
    def _scan_image_for_barcodes(self, image: np.ndarray, method: str, offset: Tuple[int, int]) -> List[Dict]:
        """Scan preprocessed image for barcodes"""
        barcodes = []
        
        try:
            # Decode barcodes
            decoded_objects = pyzbar.decode(image)
            
            for obj in decoded_objects:
                # Extract barcode data
                barcode_data = obj.data.decode('utf-8')
                barcode_type = obj.type
                
                # Get bounding box (adjust for offset)
                points = obj.polygon
                if len(points) == 4:
                    # Convert to bounding box
                    x_coords = [p.x + offset[0] for p in points]
                    y_coords = [p.y + offset[1] for p in points]
                    
                    bbox = [
                        min(x_coords), min(y_coords),
                        max(x_coords), max(y_coords)
                    ]
                else:
                    # Fallback to rect
                    rect = obj.rect
                    bbox = [
                        rect.left + offset[0], rect.top + offset[1],
                        rect.left + rect.width + offset[0], 
                        rect.top + rect.height + offset[1]
                    ]
                
                # Calculate confidence based on barcode quality
                confidence = self._calculate_barcode_confidence(obj, image)
                
                barcode_info = {
                    "data": barcode_data,
                    "type": barcode_type,
                    "bbox": bbox,
                    "confidence": confidence,
                    "detection_method": method,
                    "polygon": [[p.x + offset[0], p.y + offset[1]] for p in points]
                }
                
                barcodes.append(barcode_info)
                
        except Exception as e:
            logger.warning(f"Barcode scanning failed with method {method}: {e}")
        
        return barcodes
    
    def _calculate_barcode_confidence(self, barcode_obj, image: np.ndarray) -> float:
        """Calculate confidence score for detected barcode"""
        try:
            # Basic confidence based on barcode size and data length
            rect = barcode_obj.rect
            area = rect.width * rect.height
            data_length = len(barcode_obj.data)
            
            # Normalize area (assume reasonable barcode size range)
            area_score = min(1.0, area / 10000)  # Normalize to 0-1
            
            # Data length score (longer is often better for most barcode types)
            length_score = min(1.0, data_length / 20)
            
            # Quality score based on barcode type
            type_score = 1.0
            if barcode_obj.type in ['EAN13', 'UPCA', 'CODE128']:
                type_score = 1.0  # High confidence types
            elif barcode_obj.type in ['EAN8', 'UPCE']:
                type_score = 0.9
            else:
                type_score = 0.8
            
            # Combine scores
            confidence = (area_score * 0.3 + length_score * 0.3 + type_score * 0.4)
            
            return min(1.0, confidence)
            
        except Exception:
            return 0.5  # Default confidence
    
    def _deduplicate_barcodes(self, barcodes: List[Dict]) -> List[Dict]:
        """Remove duplicate barcode detections"""
        if not barcodes:
            return []
        
        unique_barcodes = []
        seen_combinations = set()
        
        for barcode in barcodes:
            # Create a key based on data and approximate position
            bbox = barcode["bbox"]
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Round position to handle slight variations
            position_key = (round(center_x / 10) * 10, round(center_y / 10) * 10)
            combination_key = (barcode["data"], position_key)
            
            if combination_key not in seen_combinations:
                seen_combinations.add(combination_key)
                unique_barcodes.append(barcode)
        
        # Sort by confidence (highest first)
        unique_barcodes.sort(key=lambda x: x["confidence"], reverse=True)
        
        return unique_barcodes
    
    def _validate_barcodes(self, barcodes: List[Dict]) -> List[Dict]:
        """Validate detected barcodes and add metadata"""
        validated = []
        
        for barcode in barcodes:
            # Basic validation
            if len(barcode["data"]) < 3:  # Too short
                continue
            
            if barcode["confidence"] < 0.3:  # Too low confidence
                continue
            
            # Add barcode type information
            barcode["validation"] = self._get_barcode_validation(barcode)
            
            validated.append(barcode)
        
        return validated
    
    def _get_barcode_validation(self, barcode: Dict) -> Dict:
        """Get validation information for a barcode"""
        data = barcode["data"]
        barcode_type = barcode["type"]
        
        validation = {
            "is_valid": True,
            "format": barcode_type,
            "length": len(data),
            "check_digit_valid": False,
            "product_info": None
        }
        
        try:
            # Validate specific barcode types
            if barcode_type in ["EAN13", "UPCA"]:
                validation["check_digit_valid"] = self._validate_ean13_checksum(data)
                validation["product_info"] = self._get_product_info_from_ean(data)
            
            elif barcode_type == "EAN8":
                validation["check_digit_valid"] = self._validate_ean8_checksum(data)
            
            # Add country/manufacturer info for UPC/EAN codes
            if barcode_type in ["EAN13", "UPCA", "EAN8", "UPCE"]:
                validation["country_code"] = self._get_country_code(data)
                validation["manufacturer_code"] = self._get_manufacturer_code(data)
        
        except Exception as e:
            logger.warning(f"Barcode validation failed: {e}")
            validation["is_valid"] = False
        
        return validation
    
    def _validate_ean13_checksum(self, data: str) -> bool:
        """Validate EAN-13 checksum"""
        if len(data) != 13:
            return False
        
        try:
            digits = [int(d) for d in data]
            checksum = 0
            
            for i in range(12):
                if i % 2 == 0:
                    checksum += digits[i]
                else:
                    checksum += digits[i] * 3
            
            check_digit = (10 - (checksum % 10)) % 10
            return check_digit == digits[12]
            
        except (ValueError, IndexError):
            return False
    
    def _validate_ean8_checksum(self, data: str) -> bool:
        """Validate EAN-8 checksum"""
        if len(data) != 8:
            return False
        
        try:
            digits = [int(d) for d in data]
            checksum = 0
            
            for i in range(7):
                if i % 2 == 0:
                    checksum += digits[i] * 3
                else:
                    checksum += digits[i]
            
            check_digit = (10 - (checksum % 10)) % 10
            return check_digit == digits[7]
            
        except (ValueError, IndexError):
            return False
    
    def _get_country_code(self, data: str) -> Optional[str]:
        """Get country code from EAN/UPC barcode"""
        if len(data) >= 3:
            prefix = data[:3]
            
            # Common country codes (simplified)
            country_codes = {
                "000": "US/Canada", "001": "US/Canada", "020": "Restricted",
                "030": "US/Canada", "040": "Restricted", "050": "Coupons",
                "200": "Restricted", "380": "Bulgaria", "383": "Slovenia",
                "385": "Croatia", "387": "Bosnia Herzegovina", "400": "Germany",
                "440": "Germany", "450": "Japan", "460": "Russia",
                "470": "Kyrgyzstan", "471": "Taiwan", "474": "Estonia",
                "475": "Latvia", "476": "Azerbaijan", "477": "Lithuania",
                "478": "Uzbekistan", "479": "Sri Lanka", "480": "Philippines",
                "481": "Belarus", "482": "Ukraine", "484": "Moldova",
                "485": "Armenia", "486": "Georgia", "487": "Kazakhstan",
                "489": "Hong Kong", "490": "Japan", "500": "UK",
                "539": "Ireland", "540": "Belgium/Luxembourg", "560": "Portugal",
                "569": "Iceland", "570": "Denmark", "590": "Poland",
                "594": "Romania", "599": "Hungary", "600": "South Africa",
                "608": "Bahrain", "609": "Mauritius", "611": "Morocco",
                "613": "Algeria", "615": "Nigeria", "616": "Kenya",
                "618": "Ivory Coast", "619": "Tunisia", "620": "Tanzania",
                "621": "Syria", "622": "Egypt", "624": "Libya",
                "625": "Jordan", "626": "Iran", "627": "Kuwait",
                "628": "Saudi Arabia", "629": "UAE", "640": "Finland",
                "690": "China", "729": "Israel", "730": "Sweden",
                "740": "Guatemala", "741": "El Salvador", "742": "Honduras",
                "743": "Nicaragua", "744": "Costa Rica", "745": "Panama",
                "746": "Dominican Republic", "750": "Mexico", "754": "Venezuela",
                "759": "Venezuela", "770": "Colombia", "773": "Uruguay",
                "775": "Peru", "777": "Bolivia", "778": "Argentina",
                "780": "Chile", "784": "Paraguay", "786": "Ecuador",
                "789": "Brazil", "850": "Cuba", "858": "Slovakia",
                "859": "Czech Republic", "860": "Serbia", "865": "Mongolia",
                "867": "North Korea", "868": "Turkey", "869": "Turkey",
                "870": "Netherlands", "880": "South Korea", "885": "Thailand",
                "888": "Singapore", "890": "India", "893": "Vietnam",
                "896": "Pakistan", "899": "Indonesia", "900": "Austria",
                "930": "Australia", "940": "New Zealand", "955": "Malaysia",
                "958": "Macau"
            }
            
            return country_codes.get(prefix, f"Unknown ({prefix})")
        
        return None
    
    def _get_manufacturer_code(self, data: str) -> Optional[str]:
        """Extract manufacturer code (simplified)"""
        if len(data) >= 7:
            return data[3:7]  # Typical manufacturer code position
        return None
    
    def _get_product_info_from_ean(self, data: str) -> Optional[Dict]:
        """Get basic product info from EAN (placeholder for API integration)"""
        # This would typically connect to a product database API
        # For now, return basic structure
        return {
            "ean": data,
            "found_in_database": False,
            "product_name": None,
            "brand": None,
            "category": None
        }
    
    def _save_detection_visualization(self, image: np.ndarray, barcodes: List[Dict], 
                                    original_path: str, region_bbox: Optional[List[float]]):
        """Save visualization of detected barcodes"""
        try:
            # Convert to PIL for easier drawing
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            draw = ImageDraw.Draw(pil_image)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            # Draw bounding boxes and labels
            colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
            
            for i, barcode in enumerate(barcodes):
                bbox = barcode["bbox"]
                color = colors[i % len(colors)]
                
                # Draw rectangle
                draw.rectangle(bbox, outline=color, width=3)
                
                # Draw label
                label = f"{barcode['type']}: {barcode['data'][:20]}..."
                label_bbox = draw.textbbox((bbox[0], bbox[1] - 25), label, font=font)
                draw.rectangle(label_bbox, fill=color)
                draw.text((bbox[0], bbox[1] - 25), label, fill='white', font=font)
                
                # Draw confidence
                conf_text = f"Conf: {barcode['confidence']:.2f}"
                draw.text((bbox[0], bbox[3] + 5), conf_text, fill=color, font=font)
            
            # Save visualization
            vis_path = self.temp_dir / f"barcode_detection_{Path(original_path).stem}.png"
            pil_image.save(vis_path)
            logger.info(f"Barcode detection visualization saved: {vis_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save visualization: {e}")
    
    def scan_page_images(self, page_data: Dict) -> List[Dict]:
        """Scan all images in a page for barcodes"""
        logger.info(f"Scanning page {page_data['page_number']} for barcodes")
        
        all_barcodes = []
        
        # Scan individual product images
        for image_info in page_data.get("images", []):
            image_path = image_info["path"]
            if os.path.exists(image_path):
                barcodes = self.detect_barcodes_in_image(image_path)
                
                # Add image context to barcodes
                for barcode in barcodes:
                    barcode["source_image"] = image_info["id"]
                    barcode["source_type"] = "product_image"
                
                all_barcodes.extend(barcodes)
        
        # Scan full page image
        page_image_path = page_data.get("page_image_path")
        if page_image_path and os.path.exists(page_image_path):
            barcodes = self.detect_barcodes_in_image(page_image_path)
            
            # Add page context to barcodes
            for barcode in barcodes:
                barcode["source_image"] = "full_page"
                barcode["source_type"] = "page_scan"
            
            all_barcodes.extend(barcodes)
        
        return all_barcodes
    
    def process_session(self) -> Dict:
        """Process all pages in session for barcode detection"""
        logger.info(f"Starting barcode scanning for session {self.session_id}")
        
        # Load page files
        page_files = sorted(self.input_dir.glob("page_*.json"))
        
        if not page_files:
            raise FileNotFoundError(f"No page files found in {self.input_dir}")
        
        session_results = {
            "session_id": self.session_id,
            "total_pages": len(page_files),
            "pages": [],
            "summary": {
                "total_barcodes": 0,
                "barcode_types": {},
                "avg_confidence": 0,
                "pages_with_barcodes": 0
            }
        }
        
        all_confidences = []
        
        for page_file in page_files:
            with open(page_file, 'r') as f:
                page_data = json.load(f)
            
            # Scan page for barcodes
            barcodes = self.scan_page_images(page_data)
            
            page_result = {
                "page_number": page_data["page_number"],
                "barcode_count": len(barcodes),
                "barcodes": barcodes
            }
            
            session_results["pages"].append(page_result)
            
            # Update summary
            if barcodes:
                session_results["summary"]["pages_with_barcodes"] += 1
                session_results["summary"]["total_barcodes"] += len(barcodes)
                
                for barcode in barcodes:
                    barcode_type = barcode["type"]
                    session_results["summary"]["barcode_types"][barcode_type] = \
                        session_results["summary"]["barcode_types"].get(barcode_type, 0) + 1
                    
                    all_confidences.append(barcode["confidence"])
            
            # Save individual page results
            result_file = self.output_dir / f"barcodes_{page_file.stem}.json"
            with open(result_file, 'w') as f:
                json.dump(page_result, f, indent=2)
        
        # Calculate average confidence
        if all_confidences:
            session_results["summary"]["avg_confidence"] = sum(all_confidences) / len(all_confidences)
        
        # Save session results
        session_file = self.output_dir / "barcode_scan_results.json"
        with open(session_file, 'w') as f:
            json.dump(session_results, f, indent=2)
        
        logger.info(f"Barcode scanning complete. Found {session_results['summary']['total_barcodes']} barcodes across {session_results['summary']['pages_with_barcodes']} pages")
        
        return session_results

def main():
    parser = argparse.ArgumentParser(description='Barcode Scanner - Detect and decode product barcodes')
    parser.add_argument('--session', required=True, help='Session ID')
    parser.add_argument('--image', help='Scan specific image file')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode for validation')
    
    args = parser.parse_args()
    
    try:
        scanner = BarcodeScanner(args.session)
        
        if args.image:
            # Scan single image
            barcodes = scanner.detect_barcodes_in_image(args.image)
            print(f"Found {len(barcodes)} barcode(s):")
            for barcode in barcodes:
                print(f"  {barcode['type']}: {barcode['data']} (confidence: {barcode['confidence']:.2f})")
        else:
            # Process entire session
            results = scanner.process_session()
            
            print(f"✅ Barcode scanning complete!")
            print(f"📊 Found {results['summary']['total_barcodes']} barcodes across {results['summary']['pages_with_barcodes']} pages")
            print(f"📁 Results saved to: {scanner.output_dir}/")
        
        return 0
        
    except Exception as e:
        logger.error(f"Barcode scanning failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())