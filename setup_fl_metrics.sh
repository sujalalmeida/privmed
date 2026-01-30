#!/bin/bash

# PrivMed FL Paper Metrics - Complete Setup Script
# This script sets up the entire metrics collection system

set -e  # Exit on error

echo "================================================"
echo "PrivMed FL Paper Metrics - Complete Setup"
echo "================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$SCRIPT_DIR/server"
DOCS_DIR="$SCRIPT_DIR/docs"

# Function to print status
print_status() {
    echo -e "${BLUE}➜${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Step 1: Verify environment
print_status "Step 1: Verifying environment..."

if [ ! -f "$SERVER_DIR/.env" ] && [ ! -f "$SCRIPT_DIR/.env" ]; then
    print_warning "No .env file found. Please create one with Supabase credentials."
    echo ""
    echo "Required variables:"
    echo "  SUPABASE_URL=https://your-project.supabase.co"
    echo "  SUPABASE_SERVICE_KEY=your-service-key"
    echo ""
    echo "Create .env file in either:"
    echo "  - $SCRIPT_DIR/.env"
    echo "  - $SERVER_DIR/.env"
    exit 1
fi

print_success "Environment file found"

# Step 2: Check Python dependencies
print_status "Step 2: Checking Python dependencies..."

cd "$SERVER_DIR"

python3 -c "import pandas, numpy, sklearn, supabase" 2>/dev/null || {
    print_warning "Missing Python dependencies. Installing..."
    pip install -q pandas numpy scikit-learn supabase python-dotenv
    print_success "Dependencies installed"
}

print_success "All dependencies available"

# Step 3: Verify files exist
print_status "Step 3: Verifying new files..."

files=(
    "fl_metrics.py"
    "fl_logging.py"
    "create_paper_metrics_tables.sql"
    "INTEGRATION_EXAMPLE.py"
    "scripts/train_centralized_baseline.py"
    "scripts/export_paper_metrics.py"
)

missing=0
for file in "${files[@]}"; do
    if [ ! -f "$SERVER_DIR/$file" ]; then
        print_warning "Missing: $file"
        missing=$((missing + 1))
    fi
done

if [ $missing -eq 0 ]; then
    print_success "All files present"
else
    echo ""
    echo "Error: $missing file(s) missing. Please ensure all files are created."
    exit 1
fi

# Step 4: Test metrics module
print_status "Step 4: Testing metrics module..."

cd "$SERVER_DIR"
python3 fl_metrics.py >/dev/null 2>&1 && {
    print_success "fl_metrics.py test passed"
} || {
    print_warning "fl_metrics.py test failed (this may be expected if test data is not available)"
}

# Step 5: Display database setup instructions
print_status "Step 5: Database setup instructions"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "MANUAL STEP REQUIRED: Create database tables"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Open Supabase Dashboard → SQL Editor"
echo "2. Run the following SQL file:"
echo ""
echo "   File: $SERVER_DIR/create_paper_metrics_tables.sql"
echo ""
echo "3. Verify tables created with:"
echo ""
echo "   SELECT table_name FROM information_schema.tables"
echo "   WHERE table_schema = 'public' AND table_name LIKE 'fl_%'"
echo "   ORDER BY table_name;"
echo ""
echo "Expected tables:"
echo "  • fl_centralized_baselines"
echo "  • fl_experiment_log"
echo "  • fl_lab_data_distribution"
echo "  • fl_model_predictions"
echo "  • fl_per_class_performance"
echo "  • fl_round_detailed_metrics"
echo ""
read -p "Press Enter after running the SQL script..."

# Step 6: Test database connection
print_status "Step 6: Testing database connection..."

python3 -c "
import os
import sys
from dotenv import load_dotenv
load_dotenv()

try:
    from supabase import create_client
    url = os.getenv('SUPABASE_URL') or os.getenv('VITE_SUPABASE_URL')
    key = os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_ANON_KEY')
    
    if not url or not key:
        print('Error: Supabase credentials not found in environment')
        sys.exit(1)
    
    sb = create_client(url, key)
    
    # Test query
    result = sb.table('fl_experiment_log').select('*').limit(1).execute()
    print('✓ Database connection successful')
    sys.exit(0)
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
" && {
    print_success "Database connection successful"
} || {
    print_warning "Database connection failed. Check your credentials and ensure tables exist."
    echo ""
    echo "Troubleshooting:"
    echo "  1. Verify SUPABASE_URL in .env"
    echo "  2. Verify SUPABASE_SERVICE_KEY in .env"
    echo "  3. Ensure tables were created (Step 5)"
    exit 1
}

# Step 7: Test baseline training (optional)
echo ""
read -p "Do you want to test centralized baseline training now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Step 7: Testing centralized baseline training..."
    
    # Check if training data exists
    if [ ! -f "$SERVER_DIR/data/combined_train.csv" ]; then
        print_warning "Training data not found at $SERVER_DIR/data/combined_train.csv"
        echo "Please run data preparation first:"
        echo "  cd server/scripts"
        echo "  python prepare_dataset.py"
    else
        cd "$SERVER_DIR/scripts"
        python3 train_centralized_baseline.py \
            --experiment-id baseline_test_$(date +%Y%m%d) \
            --log-to-db && {
            print_success "Baseline training successful!"
        } || {
            print_warning "Baseline training failed. Check error messages above."
        }
    fi
else
    print_status "Skipping baseline training test"
fi

# Step 8: Create output directory
print_status "Step 8: Creating output directory..."

mkdir -p "$SCRIPT_DIR/paper_data"
print_success "Created: $SCRIPT_DIR/paper_data"

# Final summary
echo ""
echo "================================================"
echo "Setup Complete!"
echo "================================================"
echo ""
echo "✅ Environment verified"
echo "✅ Dependencies installed"
echo "✅ Files verified"
echo "✅ Database connection tested"
echo "✅ Output directory created"
echo ""
echo "Next Steps:"
echo ""
echo "1. Train centralized baseline:"
echo "   cd server/scripts"
echo "   python train_centralized_baseline.py --experiment-id baseline_001 --log-to-db"
echo ""
echo "2. Integrate into FL pipeline:"
echo "   See: server/INTEGRATION_EXAMPLE.py"
echo "   Add FLMetricsLogger to your aggregation endpoint"
echo ""
echo "3. Export metrics for paper:"
echo "   cd server/scripts"
echo "   python export_paper_metrics.py --all --output-dir ../../paper_data"
echo ""
echo "4. Generate figures:"
echo "   See: docs/FL_PAPER_METRICS_GUIDE.md (Section: Generating Figures)"
echo ""
echo "Documentation:"
echo "  • Quick Start:   docs/QUICKSTART_FL_METRICS.md"
echo "  • Complete Guide: docs/FL_PAPER_METRICS_GUIDE.md"
echo "  • Integration:    server/INTEGRATION_EXAMPLE.py"
echo "  • Overview:       FL_METRICS_README.md"
echo ""
echo "Happy experimenting! 🚀"
