from typing import List, Tuple, Dict, Any

def process_results(raw: List[dict]) -> Tuple[List[dict], Dict[str, Any]]:
    if not raw:
        return [], {}

    # Normalize to {'text': str, 'chunks': List[str]} minimal shape
    results = []
    for r in raw:
        if "error" in r:
            results.append({"text": f"⚠️ {r['endpoint']}: {r['error']}", "chunks": []})
            continue
        text = r.get("text") or r.get("answer") or str(r)
        chunks = r.get("chunks") or []
        # if no chunks provided, make a trivial split for demo
        if not chunks and isinstance(text, str):
            chunks = [t.strip() for t in text.split("\n\n") if t.strip()]
        results.append({"text": text, "chunks": chunks})

    # Very simple metrics example
    metrics = {
        "results_count": len(results),
        "avg_chunk_count": round(sum(len(r.get("chunks", [])) for r in results) / len(results), 2),
    }
    return results, metrics