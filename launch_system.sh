#!/bin/bash
# Day 3 System Launcher

echo "🚀 Starting Clinical Trial Matching System - Day 3"
echo "================================================="

# Check if required files exist
if [ ! -f "processed_trials_day1.json" ]; then
    echo "❌ Error: processed_trials_day1.json not found. Please run Day 1 first."
    exit 1
fi

if [ ! -f "matching_results_day2.json" ]; then
    echo "⚠️  Warning: matching_results_day2.json not found. Run Day 2 for full evaluation."
fi

echo "✅ Starting system evaluation..."
python day3_complete_system.py

echo ""
echo "🌐 Choose how to run the system:"
echo "1. FastAPI server: python -m uvicorn day3_complete_system:app --reload --port 8000"
echo "2. Streamlit interface: streamlit run day3_complete_system.py --server.port 8501 -- --streamlit"
echo "3. Both (recommended): Run in separate terminals"
echo ""
echo "📖 Access points:"
echo "  • API documentation: http://localhost:8000/docs"
echo "  • Web interface: http://localhost:8501"
echo "  • API health check: http://localhost:8000/health"
