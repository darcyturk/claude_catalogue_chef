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
        
        for token in page_
