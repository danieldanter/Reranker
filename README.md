# Reranker Playground (Streamlit)

## Quickstart

```powershell
# From project root
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

streamlit run app.py
```

In the sidebar, paste a JSON list of endpoints to call in parallel, e.g.:

```json
["https://httpbin.org/get", "https://example.org"]
```
(These are just examples; replace with your real endpoints.)