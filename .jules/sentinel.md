## 2024-10-19 - Leaking exception strings via SSE streams
**Vulnerability:** Raw Python exception strings (`str(e)`) were being leaked to the frontend via SSE streams in `backend/main.py` (`_stream_agent_response` and `_stream_pipeline_response`).
**Learning:** Returning `str(e)` on top-level API endpoints violates the principle of failing securely, as it may expose internal paths, dependencies details, or sensitive configurations to users.
**Prevention:** Instead of logging/streaming `str(e)` directly to end users, generic fallback error messages (e.g. "An internal error occurred during processing.") should be used.
