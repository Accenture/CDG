#!/bin/bash
#
# Generate all paper figures.
#
# Usage:
#   ./generate_figures.sh              # Lazy mode: generate all figures with defaults
#   ./generate_figures.sh --all        # Same as above
#   ./generate_figures.sh --hyperparam # Only hyperparam figures
#   ./generate_figures.sh --histogram  # Only histogram figures
#   ./generate_figures.sh --subsample  # Only subsample/scaling figures
#   ./generate_figures.sh --help       # Show help
#
# Output goes to: results/fig_*/
#

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_BASE="$PROJECT_ROOT/results"

# Source config.sh for OUTPUT_BASE and other settings
source "$SCRIPT_DIR/../config.sh"

# Experiment results directory (where pickle files and cache are)
EXPERIMENT_RESULTS_DIR="${OUTPUT_BASE}"

# Default model/dataset for main paper figures
DEFAULT_MODEL="qwen32b"
DEFAULT_DATASET="aime2025"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}▶ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

show_help() {
    cat << EOF
Generate Paper Figures

Usage: $(basename "$0") [OPTIONS]

Options:
  --all, -a          Generate all figures (default if no option specified)
  --hyperparam, -h   Generate alpha-beta parameter sweep figures
  --histogram, -i    Generate correct vs wrong histogram figures
  --subsample, -s    Generate subsample/scaling analysis figures
  --cache-only, -c   Only use cached results, skip evaluation (fast mode)
  --model MODEL      Model for main paper figures (default: $DEFAULT_MODEL)
  --dataset DATASET  Dataset for main paper figures (default: $DEFAULT_DATASET)
  --no-show          Don't display figures (headless mode)
  --clean            Remove existing figures before generating
  --dry-run          Show what would be run without executing
  --help             Show this help message

Examples:
  $(basename "$0")                      # Generate all figures
  $(basename "$0") --cache-only         # Fast: only use cached data
  $(basename "$0") --hyperparam         # Only hyperparam figures
  $(basename "$0") --model deepseek8b   # Use different model for main figures
  $(basename "$0") --no-show --clean    # Headless, fresh start

Output directories:
  results/fig_hyperparam/   - Alpha-beta parameter sweep figures
  results/fig_histogram/    - Correct vs wrong distribution histograms
  results/fig_subsample/    - Scaling analysis figures

EOF
}

# =============================================================================
# Figure Generation Functions
# =============================================================================

generate_hyperparam_figures() {
    print_header "Generating Alpha-Beta Parameter Sweep Figures"

    local output_dir="$RESULTS_BASE/fig_hyperparam"
    mkdir -p "$output_dir"

    print_step "Output directory: $output_dir"
    print_step "Reading from: $EXPERIMENT_RESULTS_DIR (via PathConfig.OUTPUT_BASE)"

    # Main paper figures (one model × 4 datasets)
    print_step "Generating main paper figure (model: $MODEL)..."
    python "$SCRIPT_DIR/fig_hyperparam.py" \
        --results-dir "$EXPERIMENT_RESULTS_DIR" \
        --output-dir "$output_dir" \
        --main-paper --model "$MODEL" \
        $NO_SHOW_FLAG $CACHE_ONLY_FLAG \
        2>&1 | while read line; do echo "    $line"; done

    # Main paper figures (one dataset × 4 models)
    print_step "Generating main paper figure (dataset: $DATASET)..."
    python "$SCRIPT_DIR/fig_hyperparam.py" \
        --results-dir "$EXPERIMENT_RESULTS_DIR" \
        --output-dir "$output_dir" \
        --main-paper --dataset "$DATASET" \
        $NO_SHOW_FLAG $CACHE_ONLY_FLAG \
        2>&1 | while read line; do echo "    $line"; done

    # Full grid for appendix
    print_step "Generating appendix grid figure..."
    python "$SCRIPT_DIR/fig_hyperparam.py" \
        --results-dir "$EXPERIMENT_RESULTS_DIR" \
        --output-dir "$output_dir" \
        --no-individual \
        $NO_SHOW_FLAG $CACHE_ONLY_FLAG \
        2>&1 | while read line; do echo "    $line"; done

    # Individual figures (skip in cache-only mode for speed)
    if [[ "$CACHE_ONLY" != "true" ]]; then
        print_step "Generating individual figures..."
        python "$SCRIPT_DIR/fig_hyperparam.py" \
            --results-dir "$EXPERIMENT_RESULTS_DIR" \
            --output-dir "$output_dir" \
            $NO_SHOW_FLAG \
            2>&1 | while read line; do echo "    $line"; done
    fi

    print_success "Hyperparam figures saved to: $output_dir"
}

generate_histogram_figures() {
    print_header "Generating Histogram Figures (Correct vs Wrong)"

    local output_dir="$RESULTS_BASE/fig_histogram"
    mkdir -p "$output_dir"

    print_step "Output directory: $output_dir"

    # Main figure for specified model/dataset
    print_step "Generating histogram for $MODEL on $DATASET..."
    python "$SCRIPT_DIR/fig_histogram.py" \
        --results-dir "$EXPERIMENT_RESULTS_DIR" \
        --output-dir "$output_dir" \
        --model "$MODEL" \
        --dataset "$DATASET" \
        $NO_SHOW_FLAG \
        2>&1 | while read line; do echo "    $line"; done

    # Generate for all combinations (skip in cache-only mode)
    if [[ "$CACHE_ONLY" != "true" ]]; then
        print_step "Generating histograms for all model-dataset combinations..."
        python "$SCRIPT_DIR/fig_histogram.py" \
            --results-dir "$EXPERIMENT_RESULTS_DIR" \
            --output-dir "$output_dir" \
            --all \
            $NO_SHOW_FLAG \
            2>&1 | while read line; do echo "    $line"; done
    fi

    print_success "Histogram figures saved to: $output_dir"
}

generate_subsample_figures() {
    print_header "Generating Subsample/Scaling Analysis Figures"

    local output_dir="$RESULTS_BASE/fig_subsample"
    mkdir -p "$output_dir"

    print_step "Output directory: $output_dir"

    # Main paper figure (single row)
    print_step "Generating main paper figure (model: $MODEL)..."
    python "$SCRIPT_DIR/fig_subsample.py" \
        --results-dir "$EXPERIMENT_RESULTS_DIR" \
        --output-dir "$output_dir" \
        --main-paper --model "$MODEL" \
        --key-methods-only \
        $NO_SHOW_FLAG $CACHE_ONLY_FLAG \
        2>&1 | while read line; do echo "    $line"; done

    # Improvement plot
    print_step "Generating improvement over baseline plot..."
    python "$SCRIPT_DIR/fig_subsample.py" \
        --results-dir "$EXPERIMENT_RESULTS_DIR" \
        --output-dir "$output_dir" \
        --improvement --model "$MODEL" \
        $NO_SHOW_FLAG $CACHE_ONLY_FLAG \
        2>&1 | while read line; do echo "    $line"; done

    # Full grid for appendix (skip in cache-only mode)
    if [[ "$CACHE_ONLY" != "true" ]]; then
        print_step "Generating full grid for appendix..."
        python "$SCRIPT_DIR/fig_subsample.py" \
            --results-dir "$EXPERIMENT_RESULTS_DIR" \
            --output-dir "$output_dir" \
            --grid \
            $NO_SHOW_FLAG \
            2>&1 | while read line; do echo "    $line"; done

        # Method comparison
        print_step "Generating method comparison views..."
        python "$SCRIPT_DIR/fig_subsample.py" \
            --results-dir "$EXPERIMENT_RESULTS_DIR" \
            --output-dir "$output_dir" \
            --method-comparison --dataset "$DATASET" \
            $NO_SHOW_FLAG \
            2>&1 | while read line; do echo "    $line"; done

        # Compact heatmap
        print_step "Generating compact heatmap..."
        python "$SCRIPT_DIR/fig_subsample.py" \
            --results-dir "$EXPERIMENT_RESULTS_DIR" \
            --output-dir "$output_dir" \
            --compact \
            $NO_SHOW_FLAG \
            2>&1 | while read line; do echo "    $line"; done
    fi

    print_success "Subsample figures saved to: $output_dir"
}

clean_figures() {
    print_header "Cleaning Existing Figures"

    local dirs=("fig_hyperparam" "fig_histogram" "fig_subsample")

    for dir in "${dirs[@]}"; do
        local full_path="$RESULTS_BASE/$dir"
        if [[ -d "$full_path" ]]; then
            print_step "Removing $full_path..."
            rm -rf "$full_path"
        fi
    done

    print_success "Cleaned figure directories"
}

# =============================================================================
# Parse Arguments
# =============================================================================

# Defaults
RUN_HYPERPARAM=false
RUN_HISTOGRAM=false
RUN_SUBSAMPLE=false
RUN_ALL=false
CACHE_ONLY=false
NO_SHOW_FLAG="--no-show"
CACHE_ONLY_FLAG=""
MODEL="$DEFAULT_MODEL"
DATASET="$DEFAULT_DATASET"
CLEAN=false
DRY_RUN=false

# If no arguments, default to --all
if [[ $# -eq 0 ]]; then
    RUN_ALL=true
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        --all|-a)
            RUN_ALL=true
            shift
            ;;
        --hyperparam|-h)
            RUN_HYPERPARAM=true
            shift
            ;;
        --histogram|-i)
            RUN_HISTOGRAM=true
            shift
            ;;
        --subsample|-s)
            RUN_SUBSAMPLE=true
            shift
            ;;
        --cache-only|-c)
            CACHE_ONLY=true
            CACHE_ONLY_FLAG="--cache-only"
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --no-show)
            NO_SHOW_FLAG="--no-show"
            shift
            ;;
        --show)
            NO_SHOW_FLAG=""
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# If --all OR no specific figure type requested, enable all
if [[ "$RUN_ALL" == "true" ]] || [[ "$RUN_HYPERPARAM" == "false" && "$RUN_HISTOGRAM" == "false" && "$RUN_SUBSAMPLE" == "false" ]]; then
    RUN_ALL=true
    RUN_HYPERPARAM=true
    RUN_HISTOGRAM=true
    RUN_SUBSAMPLE=true
fi

# =============================================================================
# Main Execution
# =============================================================================

print_header "Paper Figure Generation"

echo "Configuration:"
echo "  Experiment data: $EXPERIMENT_RESULTS_DIR"
echo "  Figure output:   $RESULTS_BASE"
echo "  Model:           $MODEL"
echo "  Dataset:         $DATASET"
echo "  Cache-only:      $CACHE_ONLY"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    print_warning "DRY RUN MODE - No commands will be executed"
    echo ""
    echo "Would run:"
    [[ "$RUN_HYPERPARAM" == "true" ]] && echo "  - Hyperparam figures"
    [[ "$RUN_HISTOGRAM" == "true" ]] && echo "  - Histogram figures"
    [[ "$RUN_SUBSAMPLE" == "true" ]] && echo "  - Subsample figures"
    exit 0
fi

# Clean if requested
if [[ "$CLEAN" == "true" ]]; then
    clean_figures
fi

# Create base output directory
mkdir -p "$RESULTS_BASE"

# Track start time
START_TIME=$(date +%s)

# Generate figures
if [[ "$RUN_HYPERPARAM" == "true" ]]; then
    generate_hyperparam_figures
fi

if [[ "$RUN_HISTOGRAM" == "true" ]]; then
    generate_histogram_figures
fi

if [[ "$RUN_SUBSAMPLE" == "true" ]]; then
    generate_subsample_figures
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

# Summary
print_header "Summary"

echo "Generated figures:"
if [[ "$RUN_HYPERPARAM" == "true" ]]; then
    echo "  ✓ Hyperparam:  $RESULTS_BASE/fig_hyperparam/"
    ls -la "$RESULTS_BASE/fig_hyperparam/"*.pdf 2>/dev/null | awk '{print "      " $NF}' || true
fi
if [[ "$RUN_HISTOGRAM" == "true" ]]; then
    echo "  ✓ Histogram:   $RESULTS_BASE/fig_histogram/"
    ls -la "$RESULTS_BASE/fig_histogram/"*.pdf 2>/dev/null | awk '{print "      " $NF}' || true
fi
if [[ "$RUN_SUBSAMPLE" == "true" ]]; then
    echo "  ✓ Subsample:   $RESULTS_BASE/fig_subsample/"
    ls -la "$RESULTS_BASE/fig_subsample/"*.pdf 2>/dev/null | awk '{print "      " $NF}' || true
fi

echo ""
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo ""
print_success "All figures generated successfully!"
