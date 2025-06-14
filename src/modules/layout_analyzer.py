#!/usr/bin/env python3
"""
Layout Analyzer Module - The "AI Chef" Brain
Uses LayoutLM to understand document structure and extract entities.
"""

import json
import os
import sys
import argparse
from pathlib import Path
import logging
from typing import List, Dict, Any, Tuple
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LayoutAnalyzer:
    def __init__(self, session_id, model_name="microsoft/layoutlmv3-base"):
        self.session_id = session_id
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Entity labels for our catalog extraction
        self.label_list = [
            "O",  # Outside
            "B-PRODUCT_NAME", "I-PRODUCT_NAME",
            "B-SKU", "I-SKU", 
            "B-PRICE", "I-PRICE",
            "B-VARIATION", "I-VARIATION",
            "B-MIN_ORDER_QTY", "I-MIN_ORDER_QTY",
            "B-BARCODE_TEXT", "I-BARCODE_TEXT",
            "B-BRAND_NAME", "I-BRAND_NAME",
            "B-RELATED_PRODUCT", "I-RELATED_PRODUCT"
        ]
        
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load LayoutLM processor and model"""
        logger.info("Loading LayoutLM models...")
        
        try:
            # Check if we have a fine-tuned model
            finetuned_path = Path(f"models/layoutlm_finetuned")
            if finetuned_path.exists():
                logger.info("Loading fine-tuned model...")
                self.processor = LayoutLMv3Processor.from_pretrained(str(finetuned_path))
                self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                    str(finetuned_path),
                    num_labels=len(self.label_list),
                    id2label=self.id2label,
                    label2id=self.label2id
                )
            else:
                logger.info("Loading base model...")
                self.processor = LayoutLMv3Processor.from_pretrained(self.model_name)
                self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                    self.model_name,
                    num_labels=len(self.label_list),
                    id2label=self.id2label,
                    label2id=self.label2id
                )
            
            self.model.to(self.device)
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def normalize_bbox(self, bbox: List[float], page_width: float, page_height: float) -> List[int]:
        """Normalize bounding box to 0-1000 scale for LayoutLM"""
        x0, y0, x1, y1 = bbox
        
        # Normalize to 0-1000 scale
        norm_x0 = int((x0 / page_width) * 1000)
        norm_y0 = int((y0 / page_height) * 1000)
        norm_x1 = int((x1 / page_width) * 1000)
        norm_y1 = int((y1 / page_height) * 1000)
        
        return [norm_x0, norm_y0, norm_x1, norm_y1]
    
    def prepare_layoutlm_input(self, page_data: Dict) -> Dict:
        """Prepare input for LayoutLM model"""
        # Load page image
        image_path = page_data["page_image_path"]
        image = Image.open(image_path).convert("RGB")
        
        # Extract text and bounding boxes
        words = []
        boxes = []
        
        page_width = page_data["page_dimensions"]["width"]
        page_height = page_data["page_dimensions"]["height"]
        
        for token in page_data["text_tokens"]:
            words.append(token["text"])
            # Normalize bounding box
            norm_bbox = self.normalize_bbox(token["bbox"], page_width, page_height)
            boxes.append(norm_bbox)
        
        if not words:
            # Empty page handling
            words = ["[EMPTY]"]
            boxes = [[0, 0, 1000, 1000]]
        
        # Prepare encoding
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        )
        
        return encoding
    
    def predict_entities(self, page_data: Dict) -> List[Dict]:
        """Use LayoutLM to predict entities on a page"""
        logger.info(f"Analyzing layout for page {page_data['page_number']}")
        
        # Prepare input
        encoding = self.prepare_layoutlm_input(page_data)
        
        # Move to device
        for key, value in encoding.items():
            if isinstance(value, torch.Tensor):
                encoding[key] = value.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = torch.argmax(outputs.logits, dim=-1)
            confidences = torch.softmax(outputs.logits, dim=-1)
        
        # Process predictions
        tokens = encoding.input_ids[0].cpu().numpy()
        predictions = predictions[0].cpu().numpy()
        confidences = confidences[0].cpu().numpy()
        
        # Convert back to entities
        entities = self._process_predictions(
            page_data, tokens, predictions, confidences
        )
        
        return entities
    
    def _process_predictions(self, page_data: Dict, tokens, predictions, confidences) -> List[Dict]:
        """Process model predictions into structured entities"""
        entities = []
        current_entity = None
        
        text_tokens = page_data["text_tokens"]
        
        for i, (token_id, pred_id) in enumerate(zip(tokens, predictions)):
            if i >= len(text_tokens):
                break
                
            token_data = text_tokens[i]
            label = self.id2label[pred_id]
            confidence = float(np.max(confidences[i]))
            
            if label.startswith("B-"):
                # Begin new entity
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]  # Remove "B-" prefix
                current_entity = {
                    "type": entity_type,
                    "text": token_data["text"],
                    "bbox": token_data["bbox"].copy(),
                    "confidence": confidence,
                    "tokens": [token_data]
                }
                
            elif label.startswith("I-") and current_entity:
                # Continue entity
                entity_type = label[2:]  # Remove "I-" prefix
                if current_entity["type"] == entity_type:
                    current_entity["text"] += " " + token_data["text"]
                    # Expand bounding box
                    current_entity["bbox"] = self._merge_bboxes(
                        current_entity["bbox"], token_data["bbox"]
                    )
                    current_entity["tokens"].append(token_data)
                    # Update confidence (average)
                    current_entity["confidence"] = (
                        current_entity["confidence"] + confidence
                    ) / 2
                else:
                    # Type mismatch, start new entity
                    entities.append(current_entity)
                    current_entity = {
                        "type": entity_type,
                        "text": token_data["text"],
                        "bbox": token_data["bbox"].copy(),
                        "confidence": confidence,
                        "tokens": [token_data]
                    }
            else:
                # Outside or end of entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add final entity if exists
        if current_entity:
            entities.append(current_entity)
        
        # Post-process entities
        entities = self._post_process_entities(entities, page_data)
        
        return entities
    
    def _merge_bboxes(self, bbox1: List[float], bbox2: List[float]) -> List[float]:
        """Merge two bounding boxes into one encompassing both"""
        x0 = min(bbox1[0], bbox2[0])
        y0 = min(bbox1[1], bbox2[1])
        x1 = max(bbox1[2], bbox2[2])
        y1 = max(bbox1[3], bbox2[3])
        return [x0, y0, x1, y1]
    
    def _post_process_entities(self, entities: List[Dict], page_data: Dict) -> List[Dict]:
        """Apply heuristics to improve entity quality"""
        processed_entities = []
        
        for entity in entities:
            # Skip low confidence entities
            if entity["confidence"] < 0.5:
                continue
            
            # Apply type-specific processing
            if entity["type"] == "PRICE":
                entity = self._process_price_entity(entity)
            elif entity["type"] == "SKU":
                entity = self._process_sku_entity(entity)
            elif entity["type"] == "PRODUCT_NAME":
                entity = self._process_product_name_entity(entity)
            
            processed_entities.append(entity)
        
        return processed_entities
    
    def _process_price_entity(self, entity: Dict) -> Dict:
        """Apply price-specific processing"""
        text = entity["text"]
        
        # Clean up price text
        import re
        price_pattern = r'[\$\€\£]?\s*\d+[.,]\d{2}'
        match = re.search(price_pattern, text)
        if match:
            entity["cleaned_text"] = match.group()
            entity["numeric_value"] = self._extract_numeric_price(match.group())
        
        return entity
    
    def _process_sku_entity(self, entity: Dict) -> Dict:
        """Apply SKU-specific processing"""
        text = entity["text"].strip()
        
        # Remove common prefixes/suffixes
        prefixes = ["SKU:", "Item:", "Code:", "#"]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        entity["cleaned_text"] = text
        return entity
    
    def _process_product_name_entity(self, entity: Dict) -> Dict:
        """Apply product name specific processing"""
        text = entity["text"].strip()
        
        # Capitalize properly
        entity["cleaned_text"] = text.title()
        
        return entity
    
    def _extract_numeric_price(self, price_text: str) -> float:
        """Extract numeric value from price text"""
        import re
        numbers = re.findall(r'\d+[.,]\d{2}', price_text)
        if numbers:
            return float(numbers[0].replace(',', '.'))
        return 0.0
    
    def group_entities_into_products(self, entities: List[Dict], page_data: Dict) -> List[Dict]:
        """Group related entities into product records"""
        logger.info("Grouping entities into products...")
        
        products = []
        
        # Sort entities by vertical position
        entities.sort(key=lambda e: e["bbox"][1])  # Sort by y-coordinate
        
        # Use spatial proximity to group entities
        current_product = None
        y_threshold = 50  # pixels
        
        for entity in entities:
            if entity["type"] == "PRODUCT_NAME":
                # Start new product
                if current_product:
                    products.append(current_product)
                
                current_product = {
                    "page_number": page_data["page_number"],
                    "product_name": entity,
                    "entities": [entity],
                    "bbox": entity["bbox"].copy()
                }
            
            elif current_product:
                # Check if entity is close enough to current product
                product_y = current_product["bbox"][1]
                entity_y = entity["bbox"][1]
                
                if abs(entity_y - product_y) <= y_threshold:
                    # Add to current product
                    current_product["entities"].append(entity)
                    current_product["bbox"] = self._merge_bboxes(
                        current_product["bbox"], entity["bbox"]
                    )
                else:
                    # Too far, finish current product and start new one
                    products.append(current_product)
                    current_product = {
                        "page_number": page_data["page_number"],
                        "entities": [entity],
                        "bbox": entity["bbox"].copy()
                    }
                    if entity["type"] == "PRODUCT_NAME":
                        current_product["product_name"] = entity
        
        # Add final product
        if current_product:
            products.append(current_product)
        
        return products
    
    def analyze_page(self, page_data: Dict) -> Dict:
        """Complete analysis of a single page"""
        # Predict entities
        entities = self.predict_entities(page_data)
        
        # Group into products
        products = self.group_entities_into_products(entities, page_data)
        
        # Compile results
        analysis_result = {
            "page_number": page_data["page_number"],
            "total_entities": len(entities),
            "total_products": len(products),
            "entities": entities,
            "products": products,
            "confidence_stats": self._calculate_confidence_stats(entities)
        }
        
        return analysis_result
    
    def _calculate_confidence_stats(self, entities: List[Dict]) -> Dict:
        """Calculate confidence statistics"""
        if not entities:
            return {"avg": 0, "min": 0, "max": 0}
        
        confidences = [e["confidence"] for e in entities]
        return {
            "avg": sum(confidences) / len(confidences),
            "min": min(confidences),
            "max": max(confidences),
            "low_confidence_count": len([c for c in confidences if c < 0.7])
        }
    
    def process_session(self):
        """Process all pages in the session"""
        logger.info(f"Starting layout analysis for session {self.session_id}")
        
        # Load all page data
        pages_dir = Path(f"data/processed_pages/{self.session_id}")
        page_files = sorted(pages_dir.glob("page_*.json"))
        
        if not page_files:
            raise FileNotFoundError(f"No page files found in {pages_dir}")
        
        all_results = []
        
        for page_file in page_files:
            logger.info(f"Processing {page_file.name}")
            
            with open(page_file, 'r') as f:
                page_data = json.load(f)
            
            # Analyze page
            result = self.analyze_page(page_data)
            all_results.append(result)
            
            # Save individual result
            result_file = pages_dir / f"analysis_{page_file.stem}.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
        
        # Save combined results
        combined_file = pages_dir / "layout_analysis_results.json"
        with open(combined_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Generate summary
        summary = self._generate_summary(all_results)
        summary_file = pages_dir / "analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("Layout analysis complete")
        return all_results
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate analysis summary"""
        total_entities = sum(r["total_entities"] for r in results)
        total_products = sum(r["total_products"] for r in results)
        
        # Entity type distribution
        entity_types = {}
        for result in results:
            for entity in result["entities"]:
                entity_type = entity["type"]
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        # Overall confidence
        all_confidences = []
        for result in results:
            for entity in result["entities"]:
                all_confidences.append(entity["confidence"])
        
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        
        return {
            "total_pages": len(results),
            "total_entities": total_entities,
            "total_products": total_products,
            "entity_types": entity_types,
            "average_confidence": avg_confidence,
            "low_confidence_entities": len([c for c in all_confidences if c < 0.7]),
            "entities_per_page": total_entities / len(results) if results else 0,
            "products_per_page": total_products / len(results) if results else 0
        }

def main():
    parser = argparse.ArgumentParser(description='Layout Analyzer - AI-powered document understanding')
    parser.add_argument('--session', required=True, help='Session ID')
    parser.add_argument('--model', default="microsoft/layoutlmv3-base", help='LayoutLM model to use')
    
    args = parser.parse_args()
    
    try:
        # Create analyzer
        analyzer = LayoutAnalyzer(args.session, args.model)
        
        # Process all pages
        results = analyzer.process_session()
        
        print(f"✅ Layout analysis complete!")
        print(f"📊 Processed {len(results)} pages")
        print(f"📁 Results saved to: data/processed_pages/{args.session}/")
        
        return 0
        
    except Exception as e:
        logger.error(f"Layout analysis failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
