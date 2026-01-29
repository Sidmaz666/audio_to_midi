#!/bin/bash
# Run all tests

cd "$(dirname "$0")"

# Activate virtual environment if it exists
# if [ -d ".env" ]; then
#     source .env/Scripts/activate
# fi

# Run tests
echo "Running all tests..."
pytest tests/ -v

echo ""
echo "Running specific test suites:"
echo ""
echo "1. Feature extraction tests:"
pytest tests/test_features.py -v

echo ""
echo "2. Note conversion tests:"
pytest tests/test_notes.py -v -s

echo ""
echo "3. Integration tests:"
pytest tests/test_integration.py -v -s

echo ""
echo "4. API tests:"
pytest tests/test_api.py -v
