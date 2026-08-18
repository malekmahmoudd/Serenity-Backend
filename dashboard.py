"""
dashboard.py
============
A single self-contained HTML page at GET /dashboard.

WHY THIS EXISTS
---------------
/health returns JSON that is genuinely hard to read -- the classifier error in
particular is a 900-character Keras traceback on one line. When you are trying
to work out whether the model loaded and whether the LLM is real or falling
back to templates, you should not have to parse that by eye.

WHAT IT SHOWS
-------------
  * component status: classifier, LLM, crisis rule -- each green or red
  * the full error, formatted, when something failed to load
  * per-class thresholds as a bar chart, because a threshold that is wildly
    out of line with the others is the first thing to check when a class never
    fires
  * a live turn log: action, tokens, top-3 scores, and whether the reply came
    from the LLM or a hand-written template

No build step, no framework, no dependencies. One HTML string. This is a
developer tool, not part of the product -- do not expose it publicly, because
the turn log contains what users wrote.
"""

from __future__ import annotations

import collections
import html as _html
import json
from datetime import datetime

# Ring buffer of recent turns. In-memory and per-process, like sessions.
TURN_LOG: collections.deque = collections.deque(maxlen=60)


def record(session_id: str, action: str, text: str, labels: list,
           scores: dict, tokens: int, turn: int, llm_used: bool,
           reason: str = "") -> None:
    top3 = sorted(scores.items(), key=lambda kv: -kv[1])[:3] if scores else []
    TURN_LOG.appendleft({
        "time": datetime.now().strftime("%H:%M:%S"),
        "session": session_id[:8],
        "turn": turn,
        "action": action,
        "tokens": tokens,
        "labels": labels,
        "top3": [(k, round(v, 3)) for k, v in top3],
        "text": text[:90],
        "llm_used": llm_used,
        "reason": reason,
    })


ACTION_COLOURS = {
    "SAFETY": "#ef4444", "RESOLVE": "#22c55e", "ASK": "#3b82f6",
    "ABSTAIN": "#a855f7", "OOD": "#94a3b8",
}


def render(health: dict) -> str:
    clf = health.get("classifier", {})
    llm = health.get("llm", {})
    ready = bool(clf.get("ready"))
    llm_ok = bool(llm.get("configured"))
    thresholds = clf.get("thresholds") or {}
    labels = clf.get("labels") or []

    def pill(ok: bool, label: str, detail: str = "") -> str:
        colour = "#22c55e" if ok else "#ef4444"
        mark = "OK" if ok else "DOWN"
        return f"""
        <div class="card">
          <div class="pill" style="background:{colour}">{mark}</div>
          <div class="pill-body">
            <div class="pill-title">{_html.escape(label)}</div>
            <div class="pill-detail">{_html.escape(detail)}</div>
          </div>
        </div>"""

    status_cards = (
        pill(ready, "Classifier",
             f"{len(labels)} classes loaded" if ready else "not loaded") +
        pill(llm_ok, "LLM",
             llm.get("model", "") if llm_ok else "no API key -> templates") +
        pill(True, "Crisis rule", "rule-based, independent of the model") +
        pill(True, "Sessions", f"{health.get('active_sessions', 0)} active")
    )

    error_block = ""
    if not ready and clf.get("error"):
        err = _html.escape(str(clf["error"]))
        error_block = f"""
        <div class="panel error">
          <h2>Classifier failed to load</h2>
          <pre>{err}</pre>
        </div>"""

    # Threshold chart. A class whose bar is far from the others is the first
    # suspect when it never fires.
    bars = ""
    if thresholds:
        mx = max(thresholds.values()) or 1.0
        for name in sorted(thresholds, key=lambda k: -thresholds[k]):
            v = thresholds[name]
            pct = v / mx * 100
            bars += f"""
            <div class="bar-row">
              <div class="bar-label">{_html.escape(name)}</div>
              <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%"></div></div>
              <div class="bar-val">{v:.3f}</div>
            </div>"""
    threshold_block = f"""
        <div class="panel">
          <h2>Per-class thresholds <span class="sub">calibrated space</span></h2>
          {bars or '<p class="muted">No thresholds loaded.</p>'}
        </div>""" if thresholds else ""

    rows = ""
    for t in TURN_LOG:
        colour = ACTION_COLOURS.get(t["action"], "#64748b")
        top3 = " ".join(
            f'<span class="score">{_html.escape(k)} {v:.2f}</span>'
            for k, v in t["top3"]) or '<span class="muted">no scores</span>'
        lab = ", ".join(t["labels"]) if t["labels"] else "-"
        src = ('<span class="src live">LLM</span>' if t["llm_used"]
               else '<span class="src tmpl">template</span>')
        rows += f"""
        <tr>
          <td class="mono">{t['time']}</td>
          <td class="mono">{t['session']}</td>
          <td class="mono">t{t['turn']}</td>
          <td><span class="action" style="background:{colour}">{t['action']}</span></td>
          <td class="mono">{t['tokens']}</td>
          <td>{_html.escape(lab)}</td>
          <td>{top3}</td>
          <td class="reason">{_html.escape(t.get('reason', ''))}</td>
          <td>{src}</td>
          <td class="text">{_html.escape(t['text'])}</td>
        </tr>"""

    log_block = f"""
        <div class="panel">
          <h2>Recent turns <span class="sub">newest first, last {TURN_LOG.maxlen}</span></h2>
          <table>
            <thead><tr>
              <th>time</th><th>session</th><th>turn</th><th>action</th>
              <th>tok</th><th>labels</th><th>top 3 scores</th><th>why</th>
              <th>reply</th><th>message</th>
            </tr></thead>
            <tbody>{rows or '<tr><td colspan="10" class="muted">No turns yet. Send a message in the app.</td></tr>'}</tbody>
          </table>
        </div>"""

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Serenity · status</title>
<meta http-equiv="refresh" content="10">
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#0b1120; color:#e2e8f0;
         font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif; }}
  .wrap {{ max-width:1200px; margin:0 auto; padding:28px 20px 60px; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  .muted {{ color:#64748b; }}
  .sub {{ font-size:12px; font-weight:400; color:#64748b; margin-left:8px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
           gap:12px; margin:20px 0; }}
  .card {{ display:flex; gap:12px; align-items:center; background:#111827;
           border:1px solid #1f2937; border-radius:10px; padding:14px; }}
  .pill {{ font-size:11px; font-weight:700; color:#04111f; padding:3px 8px;
           border-radius:999px; letter-spacing:.04em; }}
  .pill-title {{ font-weight:600; }}
  .pill-detail {{ font-size:12px; color:#94a3b8; word-break:break-word; }}
  .panel {{ background:#111827; border:1px solid #1f2937; border-radius:10px;
            padding:18px; margin-bottom:16px; }}
  .panel h2 {{ font-size:14px; margin:0 0 14px; font-weight:600; }}
  .panel.error {{ border-color:#7f1d1d; background:#1a1114; }}
  pre {{ white-space:pre-wrap; word-break:break-word; font-size:12px;
         color:#fca5a5; margin:0; max-height:280px; overflow:auto; }}
  .bar-row {{ display:grid; grid-template-columns:140px 1fr 56px;
              gap:10px; align-items:center; margin-bottom:6px; }}
  .bar-label {{ font-size:12px; color:#cbd5e1; }}
  .bar-track {{ background:#1e293b; border-radius:4px; height:16px; }}
  .bar-fill {{ background:linear-gradient(90deg,#0ea5e9,#22d3ee);
               height:100%; border-radius:4px; }}
  .bar-val {{ font-size:12px; text-align:right; color:#94a3b8;
              font-family:ui-monospace,monospace; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  th {{ text-align:left; color:#64748b; font-weight:500; padding:6px 8px;
        border-bottom:1px solid #1f2937; white-space:nowrap; }}
  td {{ padding:7px 8px; border-bottom:1px solid #161f2e; vertical-align:top; }}
  .mono {{ font-family:ui-monospace,monospace; color:#94a3b8; }}
  .action {{ font-size:10px; font-weight:700; color:#04111f; padding:2px 7px;
             border-radius:999px; letter-spacing:.03em; }}
  .score {{ background:#1e293b; border-radius:4px; padding:1px 6px;
            margin-right:4px; font-family:ui-monospace,monospace; font-size:11px; }}
  .src {{ font-size:10px; padding:2px 6px; border-radius:4px; font-weight:600; }}
  .src.live {{ background:#052e16; color:#4ade80; }}
  .src.tmpl {{ background:#3f2d0a; color:#fbbf24; }}
  .text {{ color:#cbd5e1; max-width:250px; }}
  .reason {{ color:#64748b; font-size:11px; max-width:150px; }}
  a {{ color:#38bdf8; }}
</style></head><body><div class="wrap">
  <h1>Serenity</h1>
  <div class="muted">auto-refreshes every 10s &middot;
    <a href="/health">raw JSON</a> &middot; <a href="/docs">API docs</a></div>
  <div class="grid">{status_cards}</div>
  {error_block}
  {threshold_block}
  {log_block}
  <p class="muted" style="font-size:12px">
    Developer view. The turn log contains user message text -- do not expose
    this route publicly.
  </p>
</div></body></html>"""
