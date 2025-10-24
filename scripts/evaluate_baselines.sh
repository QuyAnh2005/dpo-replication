#!/bin/bash

# Configuration
INPUT_DIR="./inference_baselines"
OUTPUT_DIR="./eval_baselines"
EVAL_SCRIPT="evaluate.py"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Count total files
total_files=$(find "$INPUT_DIR" -name "*.json" -type f | wc -l)
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Batch Evaluation${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Total files to process: $total_files"
echo ""

# Counter for progress
current=0
success=0
failed=0

# Find all JSON files and process them
find "$INPUT_DIR" -name "*.json" -type f | sort | while read -r json_file; do
    current=$((current + 1))
    
    # Extract filename without extension
    filename=$(basename "$json_file" .json)
    
    # Create run name from filename
    run_name="eval_${filename}"
    
    echo -e "${BLUE}[$current/$total_files]${NC} Processing: ${GREEN}$filename${NC}"
    
    # Run evaluation
    if python "$EVAL_SCRIPT" \
        --model_responses "$json_file" \
        --run_name "$run_name" \
        --output_dir "$OUTPUT_DIR"; then
        success=$((success + 1))
        echo -e "${GREEN}✓ Success${NC}"
    else
        failed=$((failed + 1))
        echo -e "${RED}✗ Failed${NC}"
    fi
    
    echo ""
done

# Print summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Batch Evaluation Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Total processed: $total_files"
echo -e "${GREEN}Successful: $success${NC}"
echo -e "${RED}Failed: $failed${NC}"
echo "Results saved to: $OUTPUT_DIR"
echo -e "${BLUE}========================================${NC}"