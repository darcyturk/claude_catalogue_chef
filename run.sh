#!/bin/bash

# Colors and formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# Activate virtual environment
source .venv/bin/activate

# ASCII Art Header
echo -e "${BLUE}"
cat << "EOF"
 ██████╗ █████╗ ████████╗ █████╗ ██╗      ██████╗  ██████╗     ██████╗██╗  ██╗███████╗███████╗
██╔════╝██╔══██╗╚══██╔══╝██╔══██╗██║     ██╔═══██╗██╔════╝    ██╔════╝██║  ██║██╔════╝██╔════╝
██║     ███████║   ██║   ███████║██║     ██║   ██║██║  ███╗   ██║     ███████║█████╗  █████╗  
██║     ██╔══██║   ██║   ██╔══██║██║     ██║   ██║██║   ██║   ██║     ██╔══██║██╔══╝  ██╔══╝  
╚██████╗██║  ██║   ██║   ██║  ██║███████╗╚██████╔╝╚██████╔╝   ╚██████╗██║  ██║███████╗██║     
 ╚═════╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝ ╚═════╝  ╚═════╝     ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     
EOF
echo -e "${NC}"

echo -e "${BOLD}🧑‍🍳 Welcome to your PDF extraction kitchen! 🧑‍🍳${NC}"
echo ""

# Function to show progress bar
show_progress() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local completed=$((current * width / total))
    
    printf "\r["
    for ((i=0; i<completed; i++)); do printf "#"; done
    for ((i=completed; i<width; i++)); do printf "-"; done
    printf "] %d%%" $percentage
}

# Function to update state
update_state() {
    local key=$1
    local value=$2
    python3 -c "
import json
with open('state.json', 'r') as f:
    state = json.load(f)
state['$key'] = '$value'
with open('state.json', 'w') as f:
    json.dump(state, f, indent=2)
"
}

# Main menu loop
while true; do
    echo -e "${BOLD}🍽️  Main Menu 🍽️${NC}"
    echo "1. 🥘 Start New Catalog (Fresh ingredients)"
    echo "2. 🔥 Resume Cooking (Continue existing session)"
    echo "3. 👨‍🍳 Review & Correct (Taste test)"
    echo "4. 🎓 Train Chef (Learn from feedback)"
    echo "5. 📊 View Statistics"
    echo "6. 🚪 Exit Kitchen"
    echo ""
    
    read -p "👨‍🍳 What would you like to do? (1-6): " choice
    
    case $choice in
        1)
            echo -e "${YELLOW}🥘 Starting new catalog extraction...${NC}"
            
            # Get PDF input
            read -p "📁 Enter PDF file path or URL: " pdf_input
            
            # Generate session ID
            session_id=$(date +%s)
            update_state "session_id" "$session_id"
            update_state "current_step" "parsing"
            
            echo -e "${BLUE}🔪 Step 1: Parsing PDF ingredients...${NC}"
            if python3 src/modules/pdf_parser.py --input "$pdf_input" --session "$session_id"; then
                echo -e "${GREEN}✅ PDF parsing complete!${NC}"
                update_state "current_step" "layout_analysis"
                
                echo -e "${BLUE}🧠 Step 2: Analyzing layout with AI chef...${NC}"
                if python3 src/modules/layout_analyzer.py --session "$session_id"; then
                    echo -e "${GREEN}✅ Layout analysis complete!${NC}"
                    
                    echo -e "${BLUE}🔍 Step 3: Detecting barcodes...${NC}"
                    python3 src/modules/barcode_scanner.py --session "$session_id"
                    
                    echo -e "${BLUE}🌐 Step 4: Sourcing brand information...${NC}"
                    python3 src/modules/brand_info_extractor.py --session "$session_id"
                    
                    echo -e "${BLUE}📋 Step 5: Generating final recipe...${NC}"
                    python3 src/modules/data_aggregator.py --session "$session_id"
                    
                    echo -e "${GREEN}🎉 Cooking complete! Ready for tasting.${NC}"
                    update_state "current_step" "ready_for_review"
                else
                    echo -e "${RED}❌ Layout analysis failed.${NC}"
                fi
            else
                echo -e "${RED}❌ PDF parsing failed.${NC}"
            fi
            ;;
            
        2)
            echo -e "${YELLOW}🔥 Resuming cooking session...${NC}"
            current_step=$(python3 -c "import json; print(json.load(open('state.json'))['current_step'])")
            echo -e "${BLUE}📍 Resuming from step: $current_step${NC}"
            
            case $current_step in
                "parsing")
                    echo "Continuing with layout analysis..."
                    python3 src/modules/layout_analyzer.py
                    ;;
                "layout_analysis")
                    echo "Continuing with barcode detection..."
                    python3 src/modules/barcode_scanner.py
                    ;;
                # Add other resume points...
            esac
            ;;
            
        3)
            echo -e "${YELLOW}👨‍🍳 Time for taste testing!${NC}"
            python3 src/gui/main_gui.py --mode review
            ;;
            
        4)
            echo -e "${YELLOW}🎓 Training the AI chef...${NC}"
            read -p "📄 Enter path to ground truth JSON (optional): " ground_truth
            
            if [ -n "$ground_truth" ]; then
                echo -e "${BLUE}🔍 Comparing with ground truth...${NC}"
                python3 src/learning/diff_analyzer.py --ground_truth "$ground_truth"
            fi
            
            echo -e "${BLUE}🧠 Training model with feedback...${NC}"
            python3 src/learning/model_trainer.py
            ;;
            
        5)
            echo -e "${BLUE}📊 Kitchen Statistics${NC}"
            python3 -c "
import json
with open('state.json', 'r') as f:
    state = json.load(f)
print(f'Current session: {state.get(\"session_id\", \"None\")}')
print(f'Total pages processed: {state.get(\"processed_pages\", 0)}')
print(f'Corrections made: {state.get(\"corrections_made\", 0)}')
"
            ;;
            
        6)
            echo -e "${GREEN}👋 Thanks for using Catalog Chef! Happy cooking!${NC}"
            exit 0
            ;;
            
        *)
            echo -e "${RED}❌ Invalid choice. Please try again.${NC}"
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    clear
done
