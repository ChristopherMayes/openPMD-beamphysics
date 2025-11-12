# XSuite I/O Testing - Quick Start Guide

## 🎯 Overview

This guide helps you run the comprehensive test suite for the `xsuite_io` module.

---

## 📦 Installation

### Prerequisites
```bash
pip install pytest h5py numpy
```

---

## ▶️ Running Tests

### Run All Tests
```bash
python -m pytest tests/tests_xsuite/test_xsuite_io.py -v --override-ini="addopts="
```

### Run Specific Test Class
```bash
# Test only machine parameters
pytest test_xsuite_io.py::TestMachineParameters -v

# Test only wake tables
pytest test_xsuite_io.py::TestWakeTables -v

# Test only impedance tables
pytest test_xsuite_io.py::TestImpedanceTables -v
```

### Run Specific Test
```bash
pytest test_xsuite_io.py::TestMachineParameters::test_write_and_read_machine_parameters -v
```

### Run with Coverage
```bash
pip install pytest-cov
pytest test_xsuite_io.py --cov=xsuite_io --cov-report=html
```

### Quiet Mode
```bash
pytest test_xsuite_io.py -q
```

---

## 📊 Expected Output

```
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-9.0.1, pluggy-1.6.0
collected 33 items

test_xsuite_io.py::TestMachineParameters::test_write_and_read_machine_parameters PASSED [  3%]
test_xsuite_io.py::TestMachineParameters::test_write_with_file_object PASSED [  6%]
...
test_xsuite_io.py::TestIntegration::test_file_compatibility PASSED [100%]

============================== 33 passed in 0.97s ===============================
```

---

## 🧪 Test Organization

### Test Classes
1. **TestMachineParameters** (5 tests) - Machine parameter I/O
2. **TestWakeTables** (6 tests) - Wake potential I/O
3. **TestImpedanceTables** (4 tests) - Impedance table I/O
4. **TestFileValidation** (6 tests) - File structure validation
5. **TestComponentListing** (4 tests) - Component discovery
6. **TestEdgeCases** (6 tests) - Edge cases and special values
7. **TestIntegration** (2 tests) - End-to-end workflows

---

## 🔍 Test Details

### What Gets Tested

#### Machine Parameters
- ✅ Write/read cycle
- ✅ File object vs string paths
- ✅ Custom HDF5 paths
- ✅ Parameter updates
- ✅ Error handling

#### Wake Tables
- ✅ Longitudinal wakes (V/C)
- ✅ Transverse wakes (V/C/m)
- ✅ Multiple components
- ✅ Metadata preservation
- ✅ Component overwriting

#### Impedance Tables
- ✅ Complex impedance (real + imag)
- ✅ Multiple planes
- ✅ Resonator models
- ✅ Frequency-domain data

#### Edge Cases
- ✅ Empty arrays
- ✅ Single points
- ✅ Large datasets (10M points)
- ✅ NaN/Inf values
- ✅ Unicode metadata
- ✅ Concurrent reads

---

## 🐛 Troubleshooting

### Import Errors
```bash
# Make sure xsuite_io.py is in the same directory
ls -l xsuite_io.py test_xsuite_io.py
```

### Missing Dependencies
```bash
pip install pytest h5py numpy --upgrade
```

### Test Failures
```bash
# Run with full traceback
pytest test_xsuite_io.py -v --tb=long

# Run single failing test
pytest test_xsuite_io.py::TestClassName::test_name -v --tb=short
```

---

## 📈 Performance Testing

### Large Dataset Test
```bash
# This test writes/reads 10M points
pytest test_xsuite_io.py::TestEdgeCases::test_very_large_arrays -v -s
```

### Benchmark All Tests
```bash
pip install pytest-benchmark
# Add @pytest.mark.benchmark decorator to tests
pytest test_xsuite_io.py --benchmark-only
```

---

## 🎯 Test Coverage Goals

- ✅ **Line Coverage:** >95%
- ✅ **Branch Coverage:** >90%
- ✅ **Function Coverage:** 100%
- ✅ **Error Paths:** All covered

---

## 📝 Adding New Tests

### Template
```python
class TestNewFeature:
    """Tests for new feature."""
    
    def test_basic_functionality(self, tmp_path):
        """Test basic use case."""
        h5_file = tmp_path / "test.h5"
        
        # Setup
        # ...
        
        # Execute
        # ...
        
        # Assert
        assert result == expected
    
    def test_error_handling(self, tmp_path):
        """Test error conditions."""
        with pytest.raises(xsuite_io.XSuiteIOError):
            # code that should raise error
            pass
```

---

## 🚀 Continuous Integration

### GitHub Actions Example
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - run: pip install pytest h5py numpy
      - run: pytest test_xsuite_io.py -v
```

---

## 📚 Resources

- **pytest docs:** https://docs.pytest.org/
- **h5py docs:** https://docs.h5py.org/
- **openPMD standard:** https://github.com/openPMD/openPMD-standard

---

## ✅ Success Checklist

- [ ] All 33 tests pass
- [ ] No warnings in output
- [ ] Tests run in <2 seconds
- [ ] Code follows PEP 8
- [ ] Docstrings present
- [ ] Type hints complete

---

## 📞 Support

For issues or questions:
1. Check `XSUITE_IO_TEST_REPORT.md` for detailed results
2. Review test output with `-v` flag
3. Run specific failing tests with `--tb=long`

---

**Last Updated:** November 12, 2025  
**Status:** ✅ All Systems Operational
