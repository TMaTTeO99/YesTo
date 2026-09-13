import json

def _make_tool_response(success: bool, receipt: str, summary: str, details: str) -> str:
    return json.dumps(
        {"success": success, "receipt": receipt, "summary": summary, "details": details},
        ensure_ascii=False,
    )

