#!/usr/bin/env python3
"""Smoke-test report.html's logic in a headless JS engine.

Runs the page script against a stubbed DOM and Chart.js, exercising every tab
against every track, so template/regression errors surface without a browser.

    python scripts/check_report.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import quickjs

ROOT = Path(__file__).resolve().parents[1]

DOM_STUB = r"""
var __els = {};
function __mkEl(id) {
  return {
    id: id, _html: '', value: '', checked: id === 'f-valid' || id === 'runs-latest',
    textContent: '', options: [], dataset: {}, style: {},
    classList: { add: function(){}, remove: function(){}, contains: function(){ return false; } },
    getContext: function(){ return {}; },
    querySelectorAll: function(){ return []; },
    get innerHTML() { return this._html; },
    set innerHTML(v) {
      this._html = String(v);
      var opts = [], re = /<option value="([^"]*)"/g, m;
      while ((m = re.exec(this._html)) !== null) opts.push({ value: m[1] });
      if (opts.length) this.options = opts;
    },
  };
}
function __el(id) {
  if (!__els[id]) {
    __els[id] = __mkEl(id);
    __els[id].parentElement = __mkEl(id + '-parent');
  }
  return __els[id];
}
var document = {
  getElementById: function(id) { return __el(id); },
  querySelectorAll: function(){ return []; },
};
function Chart(ctx, cfg) { this.cfg = cfg; }
Chart.prototype.destroy = function(){};
"""

DRIVER = r"""
var __errors = [];
function __try(name, fn) {
  try { fn(); } catch (e) { __errors.push(name + ': ' + (e && e.message ? e.message : String(e)) + ' | ' + (e && e.stack || '')); }
}

var __tracks = [];
LATEST.forEach(function(r){ if (__tracks.indexOf(r.track) < 0) __tracks.push(r.track); });

var __pages = {
  leaderboard: renderLeaderboard, tradeoffs: renderTradeoffs, scaling: renderScaling,
  mechanisms: renderMechanisms, h2h: renderH2H, runs: renderRuns
};

var __checked = 0;
__tracks.forEach(function(track){
  [true, false].forEach(function(strict){
    document.getElementById('f-valid').checked = strict;
    state.track = track;
    state.preset = defaultPreset(track);
    state.task = ''; state.seq = ''; state.depth = '';
    __try('refreshFilters/' + track, refreshFilters);
    Object.keys(__pages).forEach(function(page){
      state.tab = page;
      __try(page + '/' + track + '/strict=' + strict, __pages[page]);
      __checked++;
    });
  });
});

// A populated leaderboard is the minimum sign that rendering really happened.
document.getElementById('f-valid').checked = true;
var __perTrack = {};
__tracks.forEach(function(track){
  state.track = track;
  state.preset = defaultPreset(track);
  state.task = ''; state.seq = ''; state.depth = '';
  refreshFilters();
  renderLeaderboard();
  var body = document.getElementById('lb-body').innerHTML;
  __perTrack[track] = {
    preset: state.preset,
    rows: (body.match(/<tr>/g) || []).length,
    note: document.getElementById('scope-note').innerHTML.replace(/\s+/g, ' ').trim()
  };
});

JSON.stringify({
  errors: __errors,
  checked: __checked,
  tracks: __tracks,
  runs: RUNS.length,
  latest: LATEST.length,
  perTrack: __perTrack,
  defaultTrack: __defaultTrack
});
"""


def extract_page_script(html: str) -> str:
    blocks = re.findall(r"<script>(.*?)</script>", html, re.S)
    if not blocks:
        raise SystemExit("no inline <script> found in report.html")
    return blocks[-1]


def main() -> int:
    html = (ROOT / "report.html").read_text()
    data_js = (ROOT / "report_data.js").read_text()
    page_js = extract_page_script(html)

    # init() touches tab wiring we do not stub; the driver drives renders directly.
    page_js = page_js.replace("\ninit();", "\n")

    ctx = quickjs.Context()
    ctx.set_memory_limit(1 << 30)
    try:
        ctx.eval(DOM_STUB)
        ctx.eval(data_js)
        ctx.eval(page_js)
        # Establish initial state the way init() would.
        ctx.eval("""
          var score = {};
          LATEST.forEach(function(r){
            if (r.preset !== defaultPreset(r.track) || r.n === 0) return;
            var base = BASELINE.get(cellKey(r));
            if (base && !atChance(base)) score[r.track] = (score[r.track]||0)+1;
          });
          var ranked = Object.keys(score).sort(function(a,b){ return score[b]-score[a]; })[0];
          state.track = ranked || 'lra';
          state.preset = defaultPreset(state.track);
          var __defaultTrack = state.track;
        """)
        result = ctx.eval(DRIVER)
    except quickjs.JSException as exc:
        print(f"FAIL: JavaScript error\n  {exc}", file=sys.stderr)
        return 1

    import json

    info = json.loads(result)
    print(f"runs={info['runs']}  latest={info['latest']}  tracks={info['tracks']}")
    print(f"render passes: {info['checked']}")
    print(f"default track: {info['defaultTrack']}")
    for track, detail in info["perTrack"].items():
        print(f"  {track:<7} [{detail['preset']}] rows={detail['rows']:<3} {detail['note']}")

    if info["errors"]:
        print(f"\nFAIL: {len(info['errors'])} render error(s)", file=sys.stderr)
        for err in info["errors"][:15]:
            print(f"  - {err}", file=sys.stderr)
        return 1
    if not any(d["rows"] for d in info["perTrack"].values()):
        print("\nFAIL: leaderboard rendered no rows on any track", file=sys.stderr)
        return 1

    print("\nOK — all pages rendered without errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
