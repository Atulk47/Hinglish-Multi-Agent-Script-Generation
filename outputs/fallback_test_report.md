# Agent 4 backend fallback regression-test evidence

**Result:** PASS — 5 tests, 0 failures, 0 errors, 0 skipped.

**Run environment:** Python 3.10.11, project `.venv`, pytest 9.2.0, Windows.

## Tests covered

1. `test_auto_backend_prefers_qwen_then_mt5_then_groq` — verifies automatic backend priority when local model folders are or are not available.
2. `test_qwen_load_failure_falls_back_to_groq_with_reason` — verifies a Qwen load failure selects Groq and records the requested backend, actual backend, and failure reason.
3. `test_qwen_smoke_test_failure_also_falls_back` — verifies Qwen generation-smoke-test failure also selects Groq and records why.
4. `test_mt5_load_failure_falls_back_and_records_reason` — verifies an mT5 initialization failure selects Groq and records why.
5. `test_successful_qwen_does_not_get_marked_as_fallback` — verifies a successful Qwen initialization remains Qwen and is not marked as fallback.

## Evidence files

- `fallback_test_evidence.txt` — verbose console transcript with each test name and PASS status.
- `fallback_test_results.xml` — JUnit XML results file (`tests=5`, `failures=0`, `errors=0`, `skipped=0`).

## Scope and limitation

Model initialization and the LLM client are mocked. These regression tests establish that the **backend-selection and fallback-control logic** behaves as expected; they do not prove that the local Qwen/mT5 weights load on this machine, that Groq is reachable, or that generated dialogue is high quality. Those require separate integration/runtime and quality evaluations.
