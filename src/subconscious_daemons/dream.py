"""
dream.py — Dream State Daemon
TRM-inspired recursive graph wandering for overnight processing.

The brain's theta (4-8 Hz) and delta (0.5-4 Hz) waves during sleep:
- Consolidate memories
- Find patterns across unrelated experiences
- Generate creative insights through free association

This daemon does the same:
1. Picks a random cluster of entity nodes from the knowledge graph
2. Asks NeMo to reflect on the cluster — find patterns, contradictions, insights
3. Feeds the reflection BACK as context for the next iteration (recursive)
4. Repeats until convergence or max depth
5. Stores the deepest insights as "dream logs" for morning review

Usage:
  python dream.py                   # run one dream cycle
  python dream.py --continuous      # run all night (stops at 7 AM)
  python dream.py --depth 5         # max recursion depth per cluster
  python dream.py --topic "video pipeline, dialogue, B-roll, A-roll, LTX"
"""

import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import time
import argparse
import logging
from datetime import datetime
from config import LOGS_DIR, REASON_MODEL
from store import load_graph
from llm import check_ollama, llm_call_with_retry, maybe_probe_higher
from heartbeat import write_heartbeat as _write_heartbeat
from dream_cluster import get_entity_cluster
from dream_store import save_dream
from postmortem import ingest_failure

DEFAULT_DEPTH      = 3       # recursion depth per cluster
CLUSTER_SIZE       = 5       # entities per dream cluster
WAKE_HOUR          = 7       # stop dreaming at 7 AM
PAUSE_BETWEEN_DREAMS = 10    # seconds between dream cycles
MAX_RETRIES        = 3       # retry LLM calls
MAX_CONSECUTIVE_FAILS = 5    # give up after this many failures in a row

# ── Logging ───────────────────────────────────────────────────────────────
LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOGS_DIR / f"dream-{datetime.now().strftime('%Y-%m-%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [dream] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)


def write_heartbeat(dream_count: int, error_count: int, status: str = "running"):
    """Convenience wrapper for the shared heartbeat writer."""
    _write_heartbeat(
        process="dream.py", status=status,
        dream_count=dream_count, errors=error_count,
    )

# ── Prompts ───────────────────────────────────────────────────────────────
DREAM_SEED_PROMPT = """You are the subconscious mind of a founder/architect. You are in a DREAM STATE — free association, pattern recognition, creative insight.

You are looking at a cluster of concepts from the founder's knowledge graph. Think deeply:
- What PATTERNS connect these seemingly unrelated things?
- What CONTRADICTIONS or TENSIONS exist between them?
- What INSIGHTS emerge when you hold all of them in mind at once?
- What QUESTIONS should the founder be asking that they haven't thought of?

Be creative. Be surprising. Make connections that conscious thinking would miss."""

DREAM_RECURSE_PROMPT = """You are dreaming deeper. Your previous dream-thought was:

{previous_reflection}

Now go DEEPER. What does this reflection reveal when you look at it from a different angle?
- Challenge your own previous insight
- Find the deeper pattern underneath
- Connect it to something unexpected
- What is the ONE THING the founder needs to know?

Think like a subconscious mind — not logical, but truthful."""

DREAM_CONVERGE_PROMPT = """You have been dreaming in {depth} layers. Here is the full dream trace:

{dream_trace}

Now CONVERGE. Synthesize everything into:
1. ONE key insight (max 2 sentences)
2. ONE action the founder should consider
3. ONE question worth exploring

Be extremely specific and actionable. No vague philosophical statements."""


def _get_recent_personal_context() -> str:
    """Read recent session logs and extract emotional/identity/episodic sections.
    Returns the most recent personal context (max 1500 chars) for dream grounding.
    """
    from datetime import timedelta
    context_parts = []
    today = datetime.now()
    
    for days_ago in range(3):  # last 3 days
        date = today - timedelta(days=days_ago)
        log_file_path = LOGS_DIR / f"{date.strftime('%Y-%m-%d')}.md"
        if not log_file_path.exists():
            continue
        try:
            text = log_file_path.read_text(encoding="utf-8", errors="ignore")
            # Extract personal sections: Emotional, Identity, Episodic
            for section_name in ["Emotional", "Identity", "Episodic", "Input"]:
                marker = f"### {section_name}"
                if marker in text:
                    idx = text.index(marker)
                    # Find next section or end
                    next_section = text.find("###", idx + len(marker))
                    if next_section == -1:
                        section_text = text[idx:idx+500]
                    else:
                        section_text = text[idx:next_section]
                    section_text = section_text.strip()
                    if len(section_text) > 20:  # skip empty sections
                        context_parts.append(f"[{date.strftime('%Y-%m-%d')}] {section_text[:300]}")
        except Exception:
            continue
    
    if not context_parts:
        return ""
    
    combined = "\n\n".join(context_parts[:6])  # max 6 sections
    return combined[:1500]




def dream_cycle(graph: dict, depth: int = DEFAULT_DEPTH, topic: str = None) -> dict:
    """One complete dream cycle: seed → recurse → converge."""

    cluster = get_entity_cluster(graph, topic=topic)
    if not cluster:
        return None

    # Format cluster for NeMo
    cluster_text = "\n".join(
        f"• [{c['type']}] {c['name']}: {c['description']}"
        + (("\n  Relations: " + " | ".join(c['relationships'][:3])) if c['relationships'] else "")
        for c in cluster
    )

    seed_names = [c['name'] for c in cluster]
    log.info(f"Dream cluster: {', '.join(seed_names)}")

    # Layer 1: Seed reflection
    seed_prompt = DREAM_SEED_PROMPT
    
    # Inject personal context from recent session logs
    personal_ctx = _get_recent_personal_context()
    if personal_ctx:
        seed_prompt += f"\n\n--- FOUNDER'S RECENT PERSONAL CONTEXT ---\nUse this to ground your pattern-finding in the founder's ACTUAL life, not abstract philosophy:\n{personal_ctx}\n--- END PERSONAL CONTEXT ---"
    
    if topic:
        seed_prompt += f"\n\nThe founder is especially stuck on: {topic}. Focus your pattern-finding on this area. What is the founder missing? What would unlock the next step?"

    reflection = llm_call_with_retry(
        [{"role": "system", "content": seed_prompt},
         {"role": "user", "content": f"Dream cluster:\n{cluster_text}"}],
        temperature=0.7
    )
    if reflection is None:
        log.error(f"Seed reflection failed after {MAX_RETRIES} retries")
        return None

    dream_trace = [{"depth": 1, "type": "seed", "reflection": reflection}]
    log.info(f"Depth 1: {reflection[:100]}...")

    # Layers 2..N: Recursive deepening
    previous = reflection
    for d in range(2, depth + 1):
        deeper = llm_call_with_retry(
            [{"role": "system", "content": DREAM_RECURSE_PROMPT.format(
                previous_reflection=previous
            )},
             {"role": "user", "content": f"Original cluster:\n{cluster_text}"}],
            temperature=0.6 + (d * 0.05)
        )
        if deeper is None:
            log.warning(f"Recursion depth {d} failed, stopping at depth {d-1}")
            break

        dream_trace.append({"depth": d, "type": "recurse", "reflection": deeper})
        log.info(f"Depth {d}: {deeper[:100]}...")
        previous = deeper

    # Convergence: synthesize
    trace_text = "\n\n".join(
        f"[Depth {t['depth']}]: {t['reflection']}" for t in dream_trace
    )
    convergence = llm_call_with_retry(
        [{"role": "system", "content": DREAM_CONVERGE_PROMPT.format(
            depth=len(dream_trace),
            dream_trace=trace_text
        )},
         {"role": "user", "content": "Converge now."}],
        temperature=0.3
    )
    if convergence is None:
        log.warning("Convergence failed, using last reflection")
        convergence = dream_trace[-1]["reflection"]

    dream_trace.append({"depth": len(dream_trace) + 1, "type": "convergence", "reflection": convergence})

    # Quality gate: reject generic convergences that don't reference any entity
    entity_refs = sum(1 for name in seed_names if name.lower() in convergence.lower())
    generic_phrases = ["embrace", "transcend", "journey", "tapestry", "interconnectedness",
                       "dynamic interplay", "holistic", "synergy"]
    generic_count = sum(1 for p in generic_phrases if p in convergence.lower())
    
    if entity_refs == 0 and generic_count >= 2:
        log.info(f"⚠️  Skipping generic convergence (0 entity refs, {generic_count} generic phrases)")
        return None
    
    log.info(f"Converged: {convergence[:120]}... (entity refs: {entity_refs})")

    return {
        "cluster": seed_names,
        "trace": dream_trace,
        "convergence": convergence,
        "timestamp": datetime.now().isoformat(),
        "depth": len(dream_trace)
    }





def main():
    parser = argparse.ArgumentParser(description="Dream State — recursive subconscious processing")
    parser.add_argument("--continuous", action="store_true", help="Run until wake hour")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="Max recursion depth")
    parser.add_argument("--cycles", type=int, default=1, help="Number of dream cycles (non-continuous)")
    parser.add_argument("--topic", type=str, default=None, help="Steer dreams toward a specific topic")
    args = parser.parse_args()

    log.info("😴 Dream State — Entering theta wave processing")
    log.info(f"   Model: {REASON_MODEL}")
    log.info(f"   Recursion depth: {args.depth}")
    if args.topic:
        log.info(f"   🎯 Topic steering: {args.topic}")
    log.info(f"   Mode: {'continuous (until ' + str(WAKE_HOUR) + ':00)' if args.continuous else f'{args.cycles} cycle(s)'}")

    # Pre-flight: check Ollama is alive
    log.info("   Checking Ollama...")
    if not check_ollama():
        log.error("❌ Ollama is not responding. Cannot dream. Exiting.")
        write_heartbeat(0, 1, status="failed_ollama_check")
        return
    log.info("   ✅ Ollama responding")

    graph = load_graph()

    # Check entity count
    entity_count = len([k for k, v in graph["nodes"].items()
                        if v.get("type") not in ("file", "community", "synthesis", None)])
    log.info(f"   Entity nodes available: {entity_count}")

    if entity_count < CLUSTER_SIZE:
        log.error(f"⚠️ Need at least {CLUSTER_SIZE} entity nodes. Run extract_knowledge.py first.")
        write_heartbeat(0, 0, status="not_enough_entities")
        return

    dream_count = 0
    error_count = 0
    consecutive_fails = 0

    if args.continuous:
        while True:
            now = datetime.now()
            if now.hour >= WAKE_HOUR and now.hour < 20:
                log.info(f"☀️ Wake hour reached ({WAKE_HOUR}:00). Stopping.")
                break

            # Check Ollama is still alive every cycle
            if consecutive_fails >= 2:
                log.warning(f"   {consecutive_fails} consecutive failures. Re-checking Ollama...")
                if not check_ollama():
                    log.error(f"   Ollama down. Waiting 30s before retry...")
                    write_heartbeat(dream_count, error_count, status="ollama_down_waiting")
                    time.sleep(30)
                    continue
                log.info("   Ollama recovered.")
                consecutive_fails = 0

            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                log.error(f"❌ {MAX_CONSECUTIVE_FAILS} consecutive failures. Giving up.")
                write_heartbeat(dream_count, error_count, status="too_many_failures")
                break

            try:
                graph = load_graph()
                dream = dream_cycle(graph, depth=args.depth, topic=args.topic)
                if dream:
                    save_dream(dream)
                    dream_count += 1
                    consecutive_fails = 0
                    maybe_probe_higher()
                    log.info(f"   Dreams completed: {dream_count}")
                else:
                    consecutive_fails += 1
                    error_count += 1
                    log.warning(f"   Dream cycle returned None (fail #{consecutive_fails})")
            except Exception as e:
                consecutive_fails += 1
                error_count += 1
                log.error(f"   Dream cycle crashed: {e} (fail #{consecutive_fails})")
                ingest_failure("dream.py", str(e), {"phase": "continuous_cycle", "consecutive_fails": consecutive_fails})

            write_heartbeat(dream_count, error_count)
            time.sleep(PAUSE_BETWEEN_DREAMS)
    else:
        for i in range(args.cycles):
            try:
                dream = dream_cycle(graph, depth=args.depth, topic=args.topic)
                if dream:
                    save_dream(dream)
                    dream_count += 1
            except Exception as e:
                error_count += 1
                log.error(f"   Dream cycle {i+1} crashed: {e}")
                ingest_failure("dream.py", str(e), {"phase": "single_cycle", "cycle": i+1})

    write_heartbeat(dream_count, error_count, status="completed")
    log.info(f"✅ Dream session complete. {dream_count} dream(s) saved, {error_count} error(s). Log: {log_file}")


if __name__ == "__main__":
    main()
