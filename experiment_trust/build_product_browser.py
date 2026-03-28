from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a static HTML browser for labeled product trust outputs."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("experiment_trust/artifacts/llm_trust_graph/phase_b_scored.csv"),
        help="Path to phase_b_scored.csv",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("experiment_trust/artifacts/llm_trust_graph/product_browser.html"),
        help="Path to output HTML",
    )
    return parser.parse_args()


def load_rows(input_csv: Path) -> list[dict]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)
    df = df.where(pd.notna(df), "")
    return df.to_dict(orient="records")


def build_html(rows: list[dict]) -> str:
    data_json = json.dumps(rows, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Product Trust Browser</title>
  <style>
    :root {{
      --bg: #f7f7f2;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #d1d5db;
      --accent: #0f766e;
      --warn: #b45309;
      --danger: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ height: 100%; }}
    body {{ margin: 0; font-family: ui-sans-serif, -apple-system, Segoe UI, Helvetica, Arial, sans-serif; background: var(--bg); color: var(--ink); overflow: hidden; }}
    .app {{ display: grid; grid-template-columns: 360px 1fr; height: 100vh; }}
    .sidebar {{ border-right: 1px solid var(--line); background: linear-gradient(180deg, #f0fdfa, #ffffff); padding: 14px; display: flex; flex-direction: column; overflow: hidden; }}
    .main {{ padding: 14px; overflow-y: auto; height: 100vh; }}
    .title {{ font-size: 20px; font-weight: 700; margin: 2px 0 10px; }}
    .sub {{ color: var(--muted); font-size: 13px; margin-bottom: 12px; }}
    .controls {{ display: grid; gap: 8px; margin-bottom: 10px; }}
    input, select, button {{ padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px; background: white; font-size: 13px; }}
    button {{ cursor: pointer; }}
    .list {{ display: grid; gap: 8px; overflow-y: auto; min-height: 0; padding-right: 4px; }}
    .card {{ border: 1px solid var(--line); background: var(--panel); border-radius: 10px; padding: 10px; cursor: pointer; }}
    .card.active {{ border-color: var(--accent); box-shadow: 0 0 0 2px #99f6e4; }}
    .cid {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 11px; color: var(--muted); }}
    .ctitle {{ font-size: 13px; margin-top: 4px; line-height: 1.3; }}
    .meta {{ font-size: 12px; color: var(--muted); margin-top: 5px; }}
    .badges {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 7px; }}
    .badge {{ font-size: 11px; padding: 2px 7px; border-radius: 999px; border: 1px solid var(--line); background: #f9fafb; }}
    .grid {{ display: grid; gap: 12px; }}
    .panel {{ border: 1px solid var(--line); border-radius: 12px; background: var(--panel); padding: 12px; }}
    .panel h3 {{ margin: 0 0 10px; font-size: 15px; }}
    .kv {{ display: grid; grid-template-columns: 200px 1fr; gap: 8px; font-size: 13px; padding: 4px 0; border-bottom: 1px dashed #e5e7eb; }}
    .kv:last-child {{ border-bottom: none; }}
    .k {{ color: var(--muted); }}
    .textblock {{ white-space: pre-wrap; font-size: 13px; line-height: 1.4; background: #fcfcfc; border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px; }}
    .two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .score-high {{ color: #065f46; font-weight: 600; }}
    .score-mid {{ color: var(--warn); font-weight: 600; }}
    .score-low {{ color: var(--danger); font-weight: 600; }}
    .footer {{ margin-top: 10px; color: var(--muted); font-size: 12px; }}
    @media (max-width: 980px) {{ .app {{ grid-template-columns: 1fr; }} .sidebar {{ border-right: none; border-bottom: 1px solid var(--line); max-height: 45vh; }} .main {{ height: auto; }} .two {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="app">
    <aside class="sidebar">
      <div class="title">Labeled Product Browser</div>
      <div id="summary" class="sub"></div>
      <div class="controls">
        <input id="search" type="text" placeholder="Search title, ID, or description" />
        <select id="sortMetric">
          <option value="trust_risk_index_graph">Sort metric: trust_risk_index_graph</option>
          <option value="phase_b_truth_likelihood_graph">Sort metric: phase_b_truth_likelihood_graph</option>
          <option value="claim_trust_score">Sort metric: claim_trust_score</option>
          <option value="signal_trust_score">Sort metric: signal_trust_score</option>
          <option value="heuristic_pressure_score">Sort metric: heuristic_pressure_score</option>
          <option value="competence_score">Sort metric: competence_score</option>
          <option value="benevolence_score">Sort metric: benevolence_score</option>
          <option value="integrity_score">Sort metric: integrity_score</option>
          <option value="predictability_score">Sort metric: predictability_score</option>
          <option value="overall_confidence">Sort metric: overall_confidence</option>
        </select>
        <select id="sortDir">
          <option value="desc">Sort direction: high to low</option>
          <option value="asc">Sort direction: low to high</option>
        </select>
        <select id="bucketFilter">
          <option value="all">Filter claim bucket: all</option>
          <option value="high">claim bucket: high</option>
          <option value="medium">claim bucket: medium</option>
          <option value="low">claim bucket: low</option>
        </select>
      </div>
      <div id="list" class="list"></div>
    </aside>
    <main class="main">
      <div class="two">
        <section class="panel">
          <h3>Product Inputs</h3>
          <div id="inputs"></div>
        </section>
        <section class="panel">
          <h3>BN-Adjusted Outputs</h3>
          <div id="bn"></div>
        </section>
      </div>
      <section class="panel" style="margin-top:12px;">
        <h3>LLM Labels</h3>
        <div id="llm"></div>
      </section>
      <section class="panel" style="margin-top:12px;">
        <h3>Rationales</h3>
        <div id="rationales"></div>
      </section>
      <div class="footer">Data source: phase_b_scored.csv</div>
    </main>
  </div>

  <script>
    const DATA = {data_json};

    const listEl = document.getElementById('list');
    const summaryEl = document.getElementById('summary');
    const searchEl = document.getElementById('search');
    const sortMetricEl = document.getElementById('sortMetric');
    const sortDirEl = document.getElementById('sortDir');
    const bucketFilterEl = document.getElementById('bucketFilter');

    const inputsEl = document.getElementById('inputs');
    const bnEl = document.getElementById('bn');
    const llmEl = document.getElementById('llm');
    const rationalesEl = document.getElementById('rationales');

    let filtered = [...DATA];
    let selectedRecordId = filtered.length ? String(filtered[0].record_id) : null;

    function fmt(v, digits=4) {{
      if (v === '' || v === null || v === undefined) return '-';
      const n = Number(v);
      if (Number.isNaN(n)) return String(v);
      return n.toFixed(digits);
    }}

    function clsScore(v) {{
      const n = Number(v);
      if (Number.isNaN(n)) return '';
      if (n >= 0.66) return 'score-high';
      if (n >= 0.33) return 'score-mid';
      return 'score-low';
    }}

    function kvRow(k, v, cls='') {{
      return `<div class="kv"><div class="k">${{k}}</div><div class="${{cls}}">${{v}}</div></div>`;
    }}

    function applyFilters() {{
      const q = searchEl.value.trim().toLowerCase();
      const bucket = bucketFilterEl.value;

      filtered = DATA.filter(row => {{
        const hay = [row.record_id, row.PRODUCT_ID, row.TITLE, row.BULLET_POINTS, row.DESCRIPTION]
          .map(x => String(x || '').toLowerCase()).join(' ');
        if (q && !hay.includes(q)) return false;
        if (bucket !== 'all' && String(row.claim_trust_bucket || '').toLowerCase() !== bucket) return false;
        return true;
      }});

      const sortMetric = sortMetricEl.value;
      const sortDir = sortDirEl.value === 'asc' ? 1 : -1;
      filtered.sort((a, b) => {{
        const av = a[sortMetric];
        const bv = b[sortMetric];
        const an = Number(av);
        const bn = Number(bv);

        if (!Number.isNaN(an) && !Number.isNaN(bn)) {{
          return sortDir * (an - bn);
        }}
        return sortDir * String(av || '').localeCompare(String(bv || ''));
      }});

      if (!filtered.find(x => String(x.record_id) === String(selectedRecordId))) {{
        selectedRecordId = filtered.length ? String(filtered[0].record_id) : null;
      }}

      renderList();
      renderDetails();
    }}

    function renderList() {{
      summaryEl.textContent = `${{filtered.length}} labeled products shown (of ${{DATA.length}} total)`;
      listEl.innerHTML = '';

      filtered.forEach(row => {{
        const id = String(row.record_id);
        const active = id === String(selectedRecordId) ? 'active' : '';
        const card = document.createElement('div');
        card.className = `card ${{active}}`;
        card.innerHTML = `
          <div class="cid">record_id: ${{id}}</div>
          <div class="ctitle">${{String(row.TITLE || '(no title)').slice(0, 130)}}</div>
          <div class="meta">product_id: ${{row.PRODUCT_ID || '-'}} | type: ${{row.PRODUCT_TYPE_ID || '-'}}</div>
          <div class="badges">
            <span class="badge">risk: ${{fmt(row.trust_risk_index_graph, 3)}}</span>
            <span class="badge">BN P(truth): ${{fmt(row.phase_b_truth_likelihood_graph, 3)}}</span>
            <span class="badge">claim: ${{row.claim_trust_bucket || '-'}}</span>
          </div>
        `;
        card.onclick = () => {{ selectedRecordId = id; renderList(); renderDetails(); }};
        listEl.appendChild(card);
      }});
    }}

    function renderDetails() {{
      const row = filtered.find(x => String(x.record_id) === String(selectedRecordId));
      if (!row) {{
        inputsEl.innerHTML = '<div class="k">No row selected.</div>';
        bnEl.innerHTML = '';
        llmEl.innerHTML = '';
        rationalesEl.innerHTML = '';
        return;
      }}

      inputsEl.innerHTML = [
        kvRow('record_id', row.record_id),
        kvRow('PRODUCT_ID', row.PRODUCT_ID),
        kvRow('PRODUCT_TYPE_ID', row.PRODUCT_TYPE_ID),
        kvRow('TITLE', String(row.TITLE || '')),
        kvRow('BULLET_POINTS', `<div class="textblock">${{String(row.BULLET_POINTS || '')}}</div>`),
        kvRow('DESCRIPTION', `<div class="textblock">${{String(row.DESCRIPTION || '')}}</div>`),
      ].join('');

      bnEl.innerHTML = [
        kvRow('phase_b_truth_likelihood_graph', fmt(row.phase_b_truth_likelihood_graph), clsScore(row.phase_b_truth_likelihood_graph)),
        kvRow('phase_b_truth_likelihood_logistic', fmt(row.phase_b_truth_likelihood_logistic), clsScore(row.phase_b_truth_likelihood_logistic)),
        kvRow('trust_risk_index_graph', fmt(row.trust_risk_index_graph), clsScore(1 - Number(row.trust_risk_index_graph || 0))),
        kvRow('trust_risk_index_logistic', fmt(row.trust_risk_index_logistic), clsScore(1 - Number(row.trust_risk_index_logistic || 0))),
        kvRow('graph_uncertainty_entropy', fmt(row.graph_uncertainty_entropy)),
      ].join('');

      llmEl.innerHTML = [
        kvRow('claim_trust_score', fmt(row.claim_trust_score), clsScore(row.claim_trust_score)),
        kvRow('claim_trust_bucket', row.claim_trust_bucket || '-'),
        kvRow('signal_trust_score', fmt(row.signal_trust_score), clsScore(row.signal_trust_score)),
        kvRow('signal_trust_bucket', row.signal_trust_bucket || '-'),
        kvRow('heuristic_pressure_score', fmt(row.heuristic_pressure_score), clsScore(1 - Number(row.heuristic_pressure_score || 0))),
        kvRow('heuristic_pressure_bucket', row.heuristic_pressure_bucket || '-'),
        kvRow('competence_score', fmt(row.competence_score), clsScore(row.competence_score)),
        kvRow('competence_bucket', row.competence_bucket || '-'),
        kvRow('benevolence_score', fmt(row.benevolence_score), clsScore(row.benevolence_score)),
        kvRow('benevolence_bucket', row.benevolence_bucket || '-'),
        kvRow('integrity_score', fmt(row.integrity_score), clsScore(row.integrity_score)),
        kvRow('integrity_bucket', row.integrity_bucket || '-'),
        kvRow('predictability_score', fmt(row.predictability_score), clsScore(row.predictability_score)),
        kvRow('predictability_bucket', row.predictability_bucket || '-'),
        kvRow('overall_confidence', fmt(row.overall_confidence), clsScore(row.overall_confidence)),
      ].join('');

      rationalesEl.innerHTML = [
        kvRow('rationale_claim', `<div class="textblock">${{String(row.rationale_claim || '')}}</div>`),
        kvRow('rationale_signal', `<div class="textblock">${{String(row.rationale_signal || '')}}</div>`),
        kvRow('rationale_pressure', `<div class="textblock">${{String(row.rationale_pressure || '')}}</div>`),
      ].join('');
    }}

    searchEl.addEventListener('input', applyFilters);
    sortMetricEl.addEventListener('change', applyFilters);
    sortDirEl.addEventListener('change', applyFilters);
    bucketFilterEl.addEventListener('change', applyFilters);

    applyFilters();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)
    html = build_html(rows)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output_html} with {len(rows)} records")


if __name__ == "__main__":
    main()
