"""
postmortem.py — Failure ingestion for the subconscious.
Turns errors into searchable, learnable knowledge instead of silent log lines.

Mirror Quality 6: Honest Post-Mortems — the system learns from its own failures.

Usage:
    from postmortem import ingest_failure
    
    try:
        risky_operation()
    except Exception as e:
        ingest_failure("dream.py", str(e), {"phase": "convergence", "cluster": cluster_names})
"""

import hashlib
import logging
from datetime import datetime
from config import DB_DIR
from store import load_graph, save_graph

log = logging.getLogger(__name__)


def ingest_failure(source: str, error: str, context: dict = None, severity: str = "error"):
    """Ingest a failure into the knowledge graph and (optionally) ChromaDB.
    
    Args:
        source: Module that failed (e.g., "dream.py", "think.py")
        error: Error message string
        context: Optional dict of additional context (phase, function, etc.)
        severity: "warning", "error", or "critical"
    
    This is best-effort — it must never crash the caller, even if
    the graph or ChromaDB is unavailable.
    """
    context = context or {}
    timestamp = datetime.now().isoformat()
    
    # 1. Add to knowledge graph as a postmortem node
    try:
        graph = load_graph()
        node_id = f"postmortem:{source}:{hashlib.md5(f'{error}:{timestamp}'.encode()).hexdigest()[:8]}"
        
        graph["nodes"][node_id] = {
            "type": "postmortem",
            "source": source,
            "error": error[:500],  # truncate very long errors  
            "severity": severity,
            "context": {k: str(v)[:200] for k, v in context.items()},
            "timestamp": timestamp,
        }
        
        # Link to source module if it exists as a node
        source_key = source.replace(".py", "")
        matching_nodes = [k for k in graph["nodes"] if source_key in k.lower() and graph["nodes"][k].get("type") != "postmortem"]
        for match in matching_nodes[:1]:
            graph["edges"].append({
                "source": node_id,
                "target": match,
                "relation": "failed_in",
                "description": f"Error in {source}: {error[:100]}",
            })
        
        save_graph(graph)
        log.debug(f"Postmortem recorded: {source} — {error[:80]}")
    except Exception as e:
        log.debug(f"Postmortem graph write failed (non-fatal): {e}")
    
    # 2. Embed into ChromaDB for vector search (best-effort)
    try:
        _embed_failure(source, error, context, timestamp)
    except Exception:
        pass  # embedding is optional — graph node is the important part


def _embed_failure(source: str, error: str, context: dict, timestamp: str):
    """Embed the failure into ChromaDB so it can be found via query.py.
    Uses a timeout to prevent hanging when Ollama is unresponsive
    (which is the exact scenario that often triggers postmortems).
    """
    import chromadb
    from llm import run_with_timeout, EMBED_CALL_TIMEOUT

    db_dir = DB_DIR
    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection("subconscious", metadata={"hnsw:space": "cosine"})

    date = datetime.now().strftime("%Y-%m-%d")
    cid = hashlib.md5(f"postmortem:{source}:{error}:{timestamp}".encode()).hexdigest()

    ctx_str = ", ".join(f"{k}={v}" for k, v in context.items()) if context else ""
    doc = f"Error in {source} ({date}): {error}. Context: {ctx_str}"

    import ollama as _ollama
    result, err = run_with_timeout(
        lambda: _ollama.embeddings(model="nomic-embed-text", prompt=doc),
        timeout=min(EMBED_CALL_TIMEOUT, 30),  # short timeout — don't block error handlers
        label="Postmortem embedding",
    )
    if err or not result:
        return  # skip embedding, graph node is enough

    collection.add(
        ids=[cid],
        embeddings=[result["embedding"]],
        documents=[doc],
        metadatas=[{
            "type": "postmortem",
            "source": source,
            "date": date,
            "path": f"postmortem/{source}/{date}",
            "region": "postmortem",
        }],
    )
