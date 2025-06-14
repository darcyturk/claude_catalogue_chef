# Intelligent PDF Extraction with Adaptive Learning: A Parabola.io-Inspired Interactive Environment

## 1. Core Architectural Philosophy: The "Controller" and Modular "Blocks"

The system is designed around a central **controller** that orchestrates a series of modular Python scripts, each acting as a specialized "block" in a data pipeline, similar to Parabola.io's visual workflow.

- **controller.sh (Orchestrator/Base Brain):**  
  Bash script serving as the primary entry point and workflow manager, defining the high-level operational flow and calling Python scripts sequentially or conditionally.
- **Python Modules ("Blocks"):**  
  Each major function (PDF parsing, layout analysis, interactive feedback, web scraping, barcode detection, JSON generation, learning updates) resides in its own script.
- **Persistent State Management:**  
  A `state.json` file tracks progress, user preferences, and learned parameters across sessions to allow pausing and resuming.
- **Temporary Data Exchange:**  
  Intermediate data is exchanged using temporary files (JSON, CSV, images) or standard IO, managed by the controller.

---

## 2. Detailed Plan for Interactive & Adaptive Extraction

### 2.1. `setup_environment.sh` (Initial Setup & Dependencies)
- **Purpose:** Automate installation of libraries and tools.
- **Implementation:**
  - Bash script to check/install Python 3, pip, git.
  - Install libraries: PyMuPDF, Pillow, Playwright, transformers (LayoutLM), datasets, opencv-python, pyzbar, fuzzywuzzy, jsondiff/recursive-diff.
  - Install Playwright browser binaries.
  - Instructions for Tesseract-OCR.
  - Create necessary directories:  
    `data/raw_pdf`, `data/processed_pages`, `data/images/products`, `data/images/brands`, `data/learning_data`, `output`.

### 2.2. `pdf_parser.py` ("PDF Input" Block)
- **Purpose:** Extract text, images, and metadata from PDF.
- **Implementation:**
  - Use PyMuPDF for extraction.
  - Output structured JSON per page.
  - Intelligent OCR triggers for image-based/sparse text pages.

### 2.3. `layout_analyzer.py` ("Intelligent Layout & Entity Recognition" Block)
- **Purpose:** Use LayoutLM to understand structure; tag product entities.
- **Implementation:**
  - Prepare LayoutLM inputs.
  - Semantic entity recognition (SKU, PRODUCT_NAME, PRICE, etc.).
  - Group NER outputs and detect columns.
  - Output `preliminary_structured_data.json`.

### 2.4. `interactive_feedback.py` ("User Feedback & Correction" Block)
- **Purpose:** Interactive CLI for review and correction.
- **Implementation:**
  - Command-line interface for reviewing/correcting entities.
  - Visual review with annotated images.
  - Guided correction and learning data collection.

### 2.5. `brand_sourcing.py` ("Web Lookup" Block)
- **Purpose:** Source brand URLs and logos.
- **Implementation:**
  - Use Playwright for Google searches.
  - Download brand logos.
  - Fuzzy string matching for brand association.

### 2.6. `barcode_detector.py` ("Barcode Scan" Block)
- **Purpose:** Detect and decode barcodes from product images/regions.
- **Implementation:**
  - Use OpenCV and pyzbar for barcode detection.
  - Interactive validation.

### 2.7. `json_generator.py` ("Output" Block)
- **Purpose:** Consolidate all data into final JSON output.
- **Implementation:**
  - Gather/organize all product and brand data.
  - Output to `output/2023-grower-catalog.json`.

### 2.8. `diff_learner.py` ("Comparison & Learning" Block)
- **Purpose:** Compare output with ground truth and generate learning data.
- **Implementation:**
  - Use a diff library to detect differences.
  - Convert corrections into new labeled data points.

### 2.9. `learning_updater.py` ("Model Training" Block)
- **Purpose:** Fine-tune LayoutLM and post-processing heuristics using feedback.
- **Implementation:**
  - Convert corrections to BIO format, fine-tune LayoutLM, refine heuristics, update state.

### 2.10. `controller.sh` (Enhanced Interactive Environment)
- **Purpose:** Provide the Parabola.io-like interactive experience.
- **Implementation:**
  - Main menu for workflow steps.
  - Guided workflow with conditional steps.
  - Visual cues, progress bars, error handling, and playful elements.

---

## 3. GitHub Project Considerations

- **Clear README:** Explain project, inspiration, setup, usage, and learning mechanism.
- **Contribution Guidelines:** Encourage contributions for heuristics, models, or UI.
- **Example Catalog:** Include sample PDF and ground truth JSON.
- **License:** Consider MIT or Apache 2.0.
- **Screenshots/GIFs:** Showcase the interactive environment.

---

This plan aims to create a robust, intelligent, and user-friendly PDF extraction tool inspired by Parabola.io—making data extraction engaging and continuously improving.
