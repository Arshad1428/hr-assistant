"""
MiniRAG Lab - preflight check
=============================
 
Run this ONCE, with internet, on the machine you will bring to the lab.
 
It downloads and caches everything the lab needs, then proves the whole
stack works end to end. After it passes, the lab runs with no network.
 
    python preflight.py
 
To confirm the offline claim afterwards (recommended - do it on a
disconnected laptop, or with wifi off):
 
    python preflight.py --offline
 
Why this exists: `pip install` and `ollama pull` can both succeed while
the lab is still not ready, because the embedding model downloads lazily
on first use. People discover this in the training room, on shared wifi,
twenty at a time.
"""
 
import argparse
import os
import shutil
import sys
from pathlib import Path
 
# Must be set before chromadb is imported - its telemetry call blocks on a
# network timeout when there is no route out, which reads as "the lab hangs".
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
 
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"
CORPUS_DIR = Path(__file__).parent / "corpus"
 
PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []
 
 
def record(status: str, name: str, detail: str = "", fix: str = "") -> bool:
    results.append((status, name, detail, fix))
    symbol = {PASS: "  ok  ", FAIL: " FAIL ", WARN: " warn "}[status]
    print(f"[{symbol}] {name}")
    if detail:
        print(f"           {detail}")
    if status != PASS and fix:
        print(f"           fix: {fix}")
    return status == PASS
 
 
# ---------------------------------------------------------------------------
 
def check_python() -> bool:
    v = sys.version_info
    if (3, 11) <= (v.major, v.minor) < (3, 14):
        return record(PASS, "Python version", f"{v.major}.{v.minor}.{v.micro}")
    return record(FAIL, "Python version", f"{v.major}.{v.minor}.{v.micro}",
                  "use Python 3.11, 3.12 or 3.13")
 
 
def check_packages() -> bool:
    import importlib
    missing = []
    for module, package in [
        ("chromadb", "chromadb"),
        ("sentence_transformers", "sentence-transformers"),
        ("langchain_text_splitters", "langchain-text-splitters"),
        ("ollama", "ollama"),
    ]:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        return record(FAIL, "Python packages", f"missing: {', '.join(missing)}",
                      "pip install -r requirements.txt")
    return record(PASS, "Python packages", "all four present")
 
 
def check_corpus() -> bool:
    files = sorted(CORPUS_DIR.glob("*.md")) if CORPUS_DIR.exists() else []
    if len(files) >= 3:
        return record(PASS, "Corpus", f"{len(files)} documents")
    return record(FAIL, "Corpus", f"found {len(files)} documents in {CORPUS_DIR}",
                  "run this from inside the minirag-lab folder")
 
 
def check_embedding_model(offline: bool) -> bool:
    """The one that catches people out. Downloads ~90 MB on first run."""
    from sentence_transformers import SentenceTransformer
    try:
        if not offline:
            print("           (first run downloads ~90 MB, please wait)")
        model = SentenceTransformer(EMBED_MODEL)
        dim = model.get_sentence_embedding_dimension()
        if dim != 384:
            return record(WARN, "Embedding model",
                          f"loaded but dimension is {dim}, expected 384")
        cache = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        return record(PASS, "Embedding model", f"{EMBED_MODEL}, {dim} dims, cached under {cache}")
    except Exception as exc:
        fix = ("connect to the internet and run this script once"
               if offline else
               f"check network access to huggingface.co ({type(exc).__name__})")
        return record(FAIL, "Embedding model", str(exc)[:160], fix)
 
 
def check_ollama_running() -> bool:
    import ollama
    try:
        listed = ollama.list()
        names = [m.get("name") or m.get("model") for m in listed.get("models", [])]
        if any(n and n.startswith(LLM_MODEL.split(":")[0]) for n in names):
            return record(PASS, "Ollama model", f"{LLM_MODEL} available")
        return record(FAIL, "Ollama model", f"installed: {names or 'none'}",
                      f"ollama pull {LLM_MODEL}")
    except Exception as exc:
        return record(FAIL, "Ollama service", str(exc)[:160],
                      "start Ollama (it must be running, not just installed)")
 
 
def check_generation() -> bool:
    import ollama
    try:
        reply = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly: ready"}],
            options={"temperature": 0, "num_predict": 10},
        )
        text = reply["message"]["content"].strip()
        return record(PASS, "Ollama generation", f"model replied: {text[:40]!r}")
    except Exception as exc:
        return record(FAIL, "Ollama generation", str(exc)[:160],
                      f"ollama run {LLM_MODEL} \"hello\"  - check it responds")
 
 
def check_end_to_end() -> bool:
    """Build a throwaway index and retrieve from it. Proves the real path."""
    import chromadb
    from chromadb.utils import embedding_functions
 
    tmp = Path(__file__).parent / ".preflight_db"
    if tmp.exists():
        shutil.rmtree(tmp)
    try:
        client = chromadb.PersistentClient(path=str(tmp))
        collection = client.create_collection(
            name="preflight",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBED_MODEL),
            metadata={"embed_model": EMBED_MODEL, "dimension": 384},
        )
        collection.add(
            ids=["a", "b"],
            documents=[
                "Casual Leave may be availed for personal exigencies and family obligations.",
                "Travel between home and the base office is not reimbursable.",
            ],
        )
        # A deliberately semantic query: no word here appears in either document.
        result = collection.query(query_texts=["time off for a wedding"], n_results=1)
        hit = result["documents"][0][0]
        dist = result["distances"][0][0]
        if "Casual Leave" in hit:
            return record(PASS, "End to end", f"semantic retrieval works (distance {dist:.3f})")
        return record(WARN, "End to end",
                      f"retrieved the wrong document (distance {dist:.3f})",
                      "the stack runs, but check the embedding model is the expected one")
    except Exception as exc:
        return record(FAIL, "End to end", str(exc)[:160],
                      "resolve the failures above first")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
 
 
# ---------------------------------------------------------------------------
 
def main() -> int:
    parser = argparse.ArgumentParser(description="MiniRAG preflight check")
    parser.add_argument("--offline", action="store_true",
                        help="assert nothing reaches the network (run with wifi off)")
    args = parser.parse_args()
 
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("\nOFFLINE MODE - nothing should reach the network.")
        print("Turn wifi off for this to mean anything.\n")
    else:
        print("\nSETUP MODE - this will download the embedding model if needed.\n")
 
    ok = check_python()
    ok &= check_packages()
    if not ok:
        print("\nStop here and fix the above - nothing else can run.\n")
        return 1
 
    check_corpus()
    check_embedding_model(args.offline)
    check_ollama_running()
    check_generation()
    check_end_to_end()
 
    failures = [r for r in results if r[0] == FAIL]
    warnings = [r for r in results if r[0] == WARN]
 
    print("\n" + "=" * 70)
    if failures:
        print(f"NOT READY - {len(failures)} check(s) failed:")
        for _, name, _, fix in failures:
            print(f"  - {name}: {fix}")
    elif args.offline:
        print("READY, OFFLINE VERIFIED.")
        print("The lab will run with no network. Keep these set in the room:")
        print("  export HF_HUB_OFFLINE=1")
        print("  export ANONYMIZED_TELEMETRY=False")
    else:
        print("READY. Everything is cached locally.")
        print("\nNow prove it: turn wifi OFF and run")
        print("  python preflight.py --offline")
        print("\nFor an air-gapped room, copy these to each machine:")
        print(f"  {Path.home() / '.ollama' / 'models'}")
        print(f"  {Path.home() / '.cache' / 'huggingface' / 'hub'}")
    if warnings:
        print(f"\n{len(warnings)} warning(s) - the lab will run, but read them.")
    print("=" * 70 + "\n")
    return 1 if failures else 0
 
 
if __name__ == "__main__":
    sys.exit(main())
