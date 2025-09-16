# utils/api_caller.py
import os
import time
import json
from typing import Dict, List, Tuple, Optional
import requests


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


class VectorServiceCaller:
    """Handles calls to both vector service endpoints (original & reranked)"""

    def __init__(self, original_port: int = None, reranked_port: int = None, host: str = None):
        host = host or _env("VECTOR_HOST", "127.0.0.1")
        original_port = original_port or int(_env("VECTOR_ORIGINAL_PORT", "8080"))
        reranked_port = reranked_port or int(_env("VECTOR_RERANKED_PORT", "8081"))

        self.original_url = f"http://{host}:{original_port}/api/fetchMedia"
        self.reranked_url = f"http://{host}:{reranked_port}/api/fetchMedia"

        # Optional health endpoints (if your services expose them)
        self.original_health = f"http://{host}:{original_port}/health"
        self.reranked_health = f"http://{host}:{reranked_port}/health"

        print(f"[VectorServiceCaller] original={self.original_url}")
        print(f"[VectorServiceCaller] reranked={self.reranked_url}")

    def _post_with_retries(self, url: str, payload: Dict, timeout: int = 30, retries: int = 2, backoff: float = 0.5) -> Tuple[Optional[requests.Response], float, Optional[str]]:
        """POST with small retry + backoff. Returns (response, time_ms, error_str)."""
        attempt = 0
        last_err = None
        while attempt <= retries:
            try:
                start = time.time()
                # requests with json= sets the Content-Type: application/json
                resp = requests.post(url, json=payload, timeout=timeout)
                ms = (time.time() - start) * 1000
                return resp, ms, None
            except Exception as e:
                last_err = str(e)
                print(f"[VectorServiceCaller] POST attempt {attempt+1}/{retries+1} to {url} failed: {last_err}")
                if attempt == retries:
                    break
                time.sleep(backoff * (2 ** attempt))
                attempt += 1
        return None, 0.0, last_err or "unknown error"

    def _normalize_documents(self, resp_json: Dict) -> List[Dict]:
        """
        Try a few common shapes to pull out document chunks.
        Adjust this if your API returns a different structure.
        """
        if not isinstance(resp_json, dict):
            return []

        # Common patterns:
        for key in ("Documents", "documents", "results", "data"):
            if key in resp_json and isinstance(resp_json[key], list):
                return resp_json[key]

        # Nested pattern, e.g. {"data": {"Documents": [...]}}
        data = resp_json.get("data")
        if isinstance(data, dict):
            for key in ("Documents", "documents", "results"):
                if key in data and isinstance(data[key], list):
                    return data[key]

        return []

    # utils/api_caller.py  (only the fetch_chunks method shown)
    def fetch_chunks(
        self,
        query: str,
        folder_ids: Optional[List[str]] = None,
        unique_titles: Optional[List[str]] = None,
        top_k: int = 10,
        use_reranker: bool = False
    ) -> Tuple[Dict, float]:
        """
        Fetch chunks from vector service.
        Returns: (response_dict, time_taken_ms)
        """
        url = self.reranked_url if use_reranker else self.original_url

        # Always include both keys; backend requires both present
        payload: Dict = {
            "prompt": query,
            "topK": top_k,
            "folderIds": folder_ids or [],          # <-- always present
            "uniqueTitles": unique_titles or [],     # <-- always present
        }

        print(f"[VectorServiceCaller] POST {url}")
        print(f"[VectorServiceCaller] Payload: {json.dumps(payload, ensure_ascii=False)}")

        resp, time_ms, err = self._post_with_retries(url, payload, timeout=30, retries=1, backoff=0.5)
        if err is not None:
            return {"error": f"Request failed: {err}"}, time_ms

        status = resp.status_code
        text_preview = (resp.text[:500] + "...") if len(resp.text) > 500 else resp.text
        print(f"[VectorServiceCaller] Status={status}")
        if status != 200:
            print(f"[VectorServiceCaller] Body (preview): {text_preview}")
            return {"error": f"Status {status}: {text_preview}"}, time_ms

        try:
            resp_json = resp.json()
        except Exception as e:
            msg = f"Invalid JSON: {e}"
            print(f"[VectorServiceCaller] {msg}")
            return {"error": msg, "raw": text_preview}, time_ms

        docs = self._normalize_documents(resp_json)
        if not docs:
            print(f"[VectorServiceCaller] No documents found; keys={list(resp_json.keys())}")
        return {"Documents": docs, "raw": resp_json}, time_ms


    # In api_caller.py, add more debugging:
    # In api_caller.py, update the fetch_both_systems method:

    def fetch_both_systems(self, query: str, folder_ids: List[str] = None, 
                        unique_titles: List[str] = None, top_k: int = 10) -> Dict:
        """Fetch from both systems in parallel"""
        
        # Fetch from original system
        original_response, original_time = self.fetch_chunks(
            query=query,
            folder_ids=folder_ids,
            unique_titles=unique_titles,
            top_k=top_k,
            use_reranker=False
        )
        
        # Fetch from reranked system
        reranked_response, reranked_time = self.fetch_chunks(
            query=query,
            folder_ids=folder_ids,
            unique_titles=unique_titles,
            top_k=top_k,
            use_reranker=True
        )
        
        # Debug: Check what we got
        print(f"Original response type: {type(original_response)}")
        if isinstance(original_response, dict):
            print(f"Original response keys: {original_response.keys()}")
        
        print(f"Reranked response type: {type(reranked_response)}")
        if isinstance(reranked_response, dict):
            print(f"Reranked response keys: {reranked_response.keys()}")
        
        # Extract chunks - YOUR API RETURNS 'Documents' not 'chunks'!
        original_chunks = []
        reranked_chunks = []
        
        if isinstance(original_response, dict):
            # Try both 'Documents' and 'chunks' keys
            original_chunks = original_response.get('Documents', original_response.get('chunks', []))
        
        if isinstance(reranked_response, dict):
            # Try both 'Documents' and 'chunks' keys
            reranked_chunks = reranked_response.get('Documents', reranked_response.get('chunks', []))
        
        print(f"Original chunks count: {len(original_chunks)}")
        print(f"Reranked chunks count: {len(reranked_chunks)}")
        
        return {
            'original': {
                'chunks': original_chunks,
                'time_ms': original_time,
                'error': original_response.get('error') if isinstance(original_response, dict) else None
            },
            'reranked': {
                'chunks': reranked_chunks,
                'time_ms': reranked_time,
                'error': reranked_response.get('error') if isinstance(reranked_response, dict) else None
            }
        }