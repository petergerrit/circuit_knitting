#!/bin/bash
# Collect system information for reproducibility comparison
# Run with: bash get_system_info.sh > system_info_$(hostname).txt

echo "=== System Information ==="
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo ""

echo "=== OS ==="
uname -a
echo ""
cat /etc/os-release | grep -E "^(PRETTY_NAME|NAME|VERSION_ID|VERSION)"
echo ""

echo "=== Python ==="
python3 --version
echo ""

echo "=== BLAS/LAPACK (via numpy) ==="
python3 -c "import numpy; numpy.show_config()" | grep -A15 "blas\|lapack" | head -20
echo ""

echo "=== libc ==="
ldd --version 2>&1 | head -1
echo ""

echo "=== libstdc++ ==="
g++ --version 2>&1 | head -1
echo ""

echo "=== GLIBC ==="
ldd --version 2>&1 | grep -i glibc | head -1
echo ""

echo "=== Qiskit Backend Info ==="
python3 -c "from qiskit_ibm_runtime.fake_provider import FakeWashingtonV2; backend = FakeWashingtonV2(); noise_model = backend._get_noise_model_from_backend_v2(); print(f'Noise model type: {type(noise_model).__name__}'); print(f'Noise model basis gates: {noise_model.basis_gates}')"
echo ""

echo "=== All pip packages ==="
pip freeze
