#!/bin/bash
# QNN Test Runner
# Sets up the QNN environment and runs conversion tests

echo "Setting up QNN environment..."

# Activate Python 3.10 virtual environment
source /tmp/qnn_env/bin/activate

# Set QNN SDK environment variables
export QNN_SDK_ROOT=/tmp/qairt/2.35.0.250530
export PYTHONPATH=${QNN_SDK_ROOT}/lib/python:${PYTHONPATH}
export LD_LIBRARY_PATH=${QNN_SDK_ROOT}/lib/x86_64-linux-clang:${LD_LIBRARY_PATH}
export PATH=${QNN_SDK_ROOT}/bin/x86_64-linux-clang:${PATH}

echo "QNN_SDK_ROOT: $QNN_SDK_ROOT"
echo "Running QNN conversion tests..."
echo ""

# Run the tests
python3.10 dextrah_lab/validation/test_qnn_conversion.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✓ QNN tests completed successfully"
else
    echo ""
    echo "✗ QNN tests failed with exit code $exit_code"
fi

exit $exit_code
