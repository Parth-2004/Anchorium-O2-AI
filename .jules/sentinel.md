## 2025-02-28 - FastAPI Swagger UI and CSP Headers
**Vulnerability:** Missing Content-Security-Policy (CSP) headers can lead to XSS attacks, but applying an overly strict `default-src 'self'` in a FastAPI backend breaks the built-in Swagger UI (`/docs`) and ReDoc (`/redoc`) pages.
**Learning:** The auto-generated documentation endpoints in FastAPI rely on inline scripts and external assets (like cdn.jsdelivr.net). An overly strict CSP blocks these, rendering the API documentation unusable.
**Prevention:** When applying CSP headers to FastAPI backends, tune the rules to allow necessary inline scripts and external assets to ensure the built-in docs continue to function while still mitigating XSS risks.
