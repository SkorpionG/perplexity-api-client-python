# Module testing instructions

## Using unittest

Run basic unit tests:

```bash
python -m unittest tests/test_module_unittest.py -v
# Or
poetry run python -m unittest tests/test_module_unittest.py -v
```

Run live unit tests:

```bash
poetry run python -m unittest tests/test_module_unittest_live.py -v
```

Run error unit tests:

```bash
poetry run python -m unittest tests/test_module_unittest_errors.py -v
```

Run error unit tests live:

```bash
poetry run python -m unittest tests/test_module_unittest_errors_live.py -v
```
