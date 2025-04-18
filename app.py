#!/usr/bin/env python
"""
Flask API for PyTorch Documentation Search Tool.
MCP‑compliant endpoint for Claude Code CLI integration.
"""

import os
import sys
import json
import time
import logging
import traceback
from typing import Dict, Any

from flask import (
    Flask,
    request,
    jsonify,
    Response,
    stream_with_context,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="flask_api.log",
)
logger = logging.getLogger("flask_api")

# ---------------------------------------------------------------------------
# Local imports: make repo modules importable
# ---------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    from scripts.search.query_processor import QueryProcessor
    from scripts.search.result_formatter import ResultFormatter
    from scripts.database.chroma_manager import ChromaManager
    from scripts.config import MAX_RESULTS, OPENAI_API_KEY  # noqa: F401  (key may be unused here)
except ImportError as e:
    logger.error("Error importing search modules: %s", e)
    raise

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# MCP tool descriptor (sent during handshake)
# ---------------------------------------------------------------------------
TOOL_DESCRIPTOR = {
    "name": "search_pytorch_docs",
    "description": (
        "Search PyTorch documentation or examples. "
        "Call when the user asks about a PyTorch API, error message, "
        "best‑practice or needs a code snippet."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "num_results": {"type": "integer", "default": 5},
            "filter": {"type": "string", "enum": ["code", "text", None]},
        },
        "required": ["query"],
    },
}

# ---------------------------------------------------------------------------
# SSE events route for MCP handshake
# ---------------------------------------------------------------------------
@app.route("/events")
def events() -> Response:
    """Server‑Sent Events stream for Claude Code MCP handshake."""

    def gen():
        # Claude expects the *first* block to describe the available tools.
        yield f"event: tools\ndata: {json.dumps([TOOL_DESCRIPTOR])}\n\n"
        # Keep‑alive comments every 15 s so the socket stays open.
        while True:
            time.sleep(15)
            yield ": keep‑alive\n\n"

    return Response(
        stream_with_context(gen()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # prevent buffering by reverse proxies
        },
    )

# ---------------------------------------------------------------------------
# Optional HTTP fallback for tools/list
# ---------------------------------------------------------------------------
@app.route("/tools/list", methods=["GET"])
def list_tools():
    """Expose tool descriptor via simple GET for HTTP transport."""
    return jsonify([TOOL_DESCRIPTOR])

# ---------------------------------------------------------------------------
# Search route (invoked by MCP \u201ccall\u201d events)
# ---------------------------------------------------------------------------
@app.route("/search", methods=["POST"])
def search():
    """Handle search requests from Claude Code.

    Expected JSON body::

        {
            "query": "string",          # required
            "num_results": 5,           # optional
            "filter": "code"            # optional
        }
    """

    request_start = time.time()
    try:
        payload = request.get_json(force=True, silent=False)
    except Exception:
        logger.warning("Malformed JSON in request body")
        return jsonify({"error": "invalid JSON"}), 400

    query = payload.get("query") if isinstance(payload, dict) else None
    if not isinstance(query, str) or not query.strip():
        return (
            jsonify({"error": "'query' field (string) is required"}),
            400,
        )

    num_results = int(payload.get("num_results", MAX_RESULTS))
    filter_type = payload.get("filter")

    try:
        results = search_pytorch_docs(query, num_results, filter_type)
        results.setdefault("timing", {})["total_request"] = time.time() - request_start
        return jsonify(results)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Unhandled error: %s", e, exc_info=True)
        return (
            jsonify(
                {
                    "error": f"internal server error: {e}",
                    "status": {"complete": False, "error_type": "server_exception"},
                }
            ),
            500,
        )

# ---------------------------------------------------------------------------
# Core search function (wraps project components)
# ---------------------------------------------------------------------------

def search_pytorch_docs(query: str, num_results: int, filter_type: str | None) -> Dict[str, Any]:
    """Run semantic search over the embedded PyTorch docs."""

    timing: Dict[str, float] = {}
    overall_start = time.time()

    # Initialise heavy components once and reuse (singleton pattern)
    global _QP, _RF, _DB  # type: ignore  # noqa: SL001
    if "_QP" not in globals():
        _QP = QueryProcessor()
        _RF = ResultFormatter()
        _DB = ChromaManager()

    # Stage 1: query processing / embedding
    stage_start = time.time()
    qdata = _QP.process_query(query)
    timing["query_processing"] = time.time() - stage_start

    # Stage 2: vector search
    stage_start = time.time()
    filters = {"chunk_type": filter_type} if filter_type else None
    raw = _DB.query(qdata["embedding"], n_results=num_results, filters=filters)
    timing["database_search"] = time.time() - stage_start

    # Stage 3: format & rank
    stage_start = time.time()
    formatted = _RF.format_results(raw, query)
    ranked = _RF.rank_results(formatted, qdata["is_code_query"], qdata.get("intent_confidence", 0.75))
    timing["result_formatting"] = time.time() - stage_start

    ranked["status"] = {"complete": True}
    ranked["timing"] = timing | {"overall": time.time() - overall_start}
    return ranked

# ---------------------------------------------------------------------------
# Dev server entry‑point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== PyTorch Documentation Search API ===")
    print("SSE events: http://localhost:5000/events")
    print("Search    : http://localhost:5000/search")
    print("\nRegister with Claude Code CLI:")
    print("  claude mcp add --transport sse pytorch_search http://localhost:5000/events\n")

    app.run(host="::", port=5000, debug=False)
