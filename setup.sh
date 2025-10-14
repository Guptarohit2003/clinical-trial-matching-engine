#!/bin/bash
# Clinical Trial Semantic Matching Engine - Complete Setup Script

echo "🚀 Clinical Trial Semantic Matching Engine - Complete Setup"
echo "=========================================================="

# Check Python version
python_version=$(python --version 2>&1)
echo "✅ Python version: $python_version"

# Install requirements
echo "📦 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
echo "🧠 Downloading spaCy NLP model..."
python -m spacy download en_core_web_sm

# Verify installation
echo "🔍 Verifying installation..."
python -c "import pandas, numpy, sklearn, fuzzywuzzy, fastapi, streamlit, plotly; print('✅ All packages installed successfully')"

# Run system validation
echo "🧪 Running system validation..."
echo "Running Day 1..."
python advanced_day1_implementation.py > day1_output.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Day 1 completed successfully"
else
    echo "❌ Day 1 failed - check day1_output.log"
fi

echo "Running Day 2..."
python advanced_day2_implementation.py > day2_output.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Day 2 completed successfully"
else
    echo "❌ Day 2 failed - check day2_output.log"
fi

echo "Running Day 3 system check..."
python advanced_day3_implementation.py > day3_output.log 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Day 3 system check passed"
else
    echo "❌ Day 3 failed - check day3_output.log"
fi

echo ""
echo "🎉 Setup Complete!"
echo ""
echo "🌐 To start your system:"
echo "  1. API Server: uvicorn advanced_day3_implementation:app --reload --port 8000"
echo "  2. Web Interface: streamlit run advanced_day3_implementation.py --server.port 8501 -- --streamlit"
echo ""
echo "🔗 Access points:"
echo "  • Web Interface: http://localhost:8501"
echo "  • API Documentation: http://localhost:8000/docs"
echo ""
echo "📚 Documentation available in:"
echo "  • IMPLEMENTATION_CHECKLIST.md"
echo "  • COMPLETE_SYSTEM_DOCUMENTATION.md"
echo "  • PROJECT_COMPLETION_SUMMARY.md"
echo ""
echo "🏆 Your Clinical Trial Semantic Matching Engine is ready!"
