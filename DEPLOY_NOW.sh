#!/bin/bash

# ============================================================================
# Allocation Decision Framework - Automatic Deployment Script
# ============================================================================
# 
# This script automates the process of setting up and deploying the
# allocation-decision-framework to GitHub.
#
# Usage: bash DEPLOY_NOW.sh /path/to/repo
# 
# ============================================================================

set -e  # Exit on any error

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get target directory from argument
TARGET_DIR="${1:-.}"

echo -e "${BLUE}================================================================================${NC}"
echo -e "${BLUE}Allocation Decision Framework - Deployment Script${NC}"
echo -e "${BLUE}================================================================================${NC}"
echo ""
echo -e "${YELLOW}Target directory:${NC} $TARGET_DIR"
echo ""

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo -e "${YELLOW}Creating directory: $TARGET_DIR${NC}"
    mkdir -p "$TARGET_DIR"
fi

echo -e "${BLUE}Step 1: Creating directory structure...${NC}"
mkdir -p "$TARGET_DIR"/{allocation_framework/{models,losses,metrics,utils},tests,docs,examples}
echo -e "${GREEN}✓ Directory structure created${NC}"
echo ""

echo -e "${BLUE}Step 2: Copying root-level files...${NC}"
cp /home/claude/allocation-decision-framework-README.md "$TARGET_DIR/README.md"
cp /home/claude/LICENSE "$TARGET_DIR/"
cp /home/claude/setup.py "$TARGET_DIR/"
cp /home/claude/requirements.txt "$TARGET_DIR/"
cp /home/claude/CONTRIBUTING.md "$TARGET_DIR/"
cp /home/claude/.gitignore "$TARGET_DIR/"
echo -e "${GREEN}✓ Root files copied${NC}"
echo ""

echo -e "${BLUE}Step 3: Copying package code...${NC}"
cp /home/claude/allocation_framework/__init__.py "$TARGET_DIR/allocation_framework/"
mkdir -p "$TARGET_DIR/allocation_framework/models"
cp /home/claude/allocation_framework/models/__init__.py "$TARGET_DIR/allocation_framework/models/"
cp /home/claude/allocation_framework/models/base.py "$TARGET_DIR/allocation_framework/models/"
mkdir -p "$TARGET_DIR/allocation_framework/losses"
cp /home/claude/allocation_framework/losses/__init__.py "$TARGET_DIR/allocation_framework/losses/"
cp /home/claude/allocation_framework/losses/fairness.py "$TARGET_DIR/allocation_framework/losses/"
mkdir -p "$TARGET_DIR/allocation_framework/metrics"
cp /home/claude/allocation_framework/metrics/__init__.py "$TARGET_DIR/allocation_framework/metrics/"
cp /home/claude/allocation_framework/metrics/fairness.py "$TARGET_DIR/allocation_framework/metrics/"
echo -e "${GREEN}✓ Package code copied${NC}"
echo ""

echo -e "${BLUE}Step 4: Copying tests...${NC}"
touch "$TARGET_DIR/tests/__init__.py"
cp /home/claude/tests/test_all.py "$TARGET_DIR/tests/"
echo -e "${GREEN}✓ Tests copied${NC}"
echo ""

echo -e "${BLUE}Step 5: Copying documentation...${NC}"
cp /home/claude/docs/INSTALL.md "$TARGET_DIR/docs/"
cp /home/claude/docs/ARCHITECTURE.md "$TARGET_DIR/docs/"
cp /home/claude/docs/API.md "$TARGET_DIR/docs/"
echo -e "${GREEN}✓ Documentation copied${NC}"
echo ""

echo -e "${BLUE}Step 6: Copying examples...${NC}"
cp /home/claude/examples/01_introduction.py "$TARGET_DIR/examples/"
echo -e "${GREEN}✓ Examples copied${NC}"
echo ""

echo -e "${BLUE}Step 7: Verifying installation...${NC}"
cd "$TARGET_DIR"
python -c "import sys; sys.path.insert(0, '.'); import allocation_framework; print(f'✓ Import successful: allocation_framework v{allocation_framework.__version__}')"
echo ""

echo -e "${BLUE}Step 8: Running quick test...${NC}"
python -c "
from allocation_framework import AllocationModel, allocation_parity
import numpy as np

X = np.random.randn(20, 5)
y = np.random.binomial(1, 0.5, 20)
groups = np.random.binomial(1, 0.5, 20)

model = AllocationModel(fairness_weight=0.3)
model.fit(X, y, groups=groups)
results = model.evaluate(X, y, groups=groups)

print(f'✓ Quick test passed!')
print(f'  Accuracy: {results.predictive_accuracy:.3f}')
print(f'  Fairness: {results.fairness_score:.3f}')
"
echo ""

echo -e "${GREEN}================================================================================${NC}"
echo -e "${GREEN}✅ DEPLOYMENT COMPLETE!${NC}"
echo -e "${GREEN}================================================================================${NC}"
echo ""
echo "Next steps:"
echo "1. cd $TARGET_DIR"
echo "2. git init"
echo "3. git remote add origin https://github.com/Rifa-111/allocation-decision-framework.git"
echo "4. git add ."
echo "5. git commit -m 'Initial commit: allocation-decision-framework v0.1.0'"
echo "6. git push origin main"
echo ""
echo "Optional:"
echo "• Run full test suite: pytest tests/ -v --cov=allocation_framework"
echo "• Install for development: pip install -e '.[dev]'"
echo "• Convert example to notebook: jupyter nbconvert --to notebook examples/01_introduction.py"
echo ""
echo -e "${GREEN}Happy open-sourcing! 🚀${NC}"
