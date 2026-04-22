from __future__ import annotations

from html import escape

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse


router = APIRouter(tags=["exports"])


@router.get("/dashboard", include_in_schema=False, response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    payload = request.app.state.service.get_dashboard_data()
    jobs = payload["jobs"]
    models = payload["models"]

    job_rows = "\n".join(
        (
            "<tr>"
            f"<td>{escape(str(job['job_id']))}</td>"
            f"<td>{escape(str(job['operation']))}</td>"
            f"<td>{escape(str(job['state']))}</td>"
            f"<td>{escape(str(job['updated_at']))}</td>"
            f"<td><code>{escape(str(job['config_path']))}</code></td>"
            "</tr>"
        )
        for job in jobs
    ) or "<tr><td colspan='5'>No jobs recorded yet.</td></tr>"

    model_rows = "\n".join(
        (
            "<tr>"
            f"<td>{escape(str(model['model_key']))}</td>"
            f"<td>{escape(str(model['latest_version_id']))}</td>"
            f"<td>{escape(str(model['version_count']))}</td>"
            f"<td><code>{escape(str(model['latest_descriptor_path']))}</code></td>"
            "</tr>"
        )
        for model in models
    ) or "<tr><td colspan='4'>No exported models registered yet.</td></tr>"

    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Tensor Training Core Dashboard</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #f4f1e8;
        --card: #fffdf8;
        --ink: #1d1b19;
        --muted: #6b655d;
        --line: #d8d0c4;
        --accent: #006d77;
      }}
      body {{
        margin: 0;
        font-family: "IBM Plex Sans", "Noto Sans", sans-serif;
        background: linear-gradient(180deg, #efe7d8 0%, var(--bg) 100%);
        color: var(--ink);
      }}
      main {{
        max-width: 1200px;
        margin: 0 auto;
        padding: 32px 20px 48px;
      }}
      .hero {{
        display: grid;
        gap: 12px;
        margin-bottom: 24px;
      }}
      .summary {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-bottom: 24px;
      }}
      .card {{
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 8px 24px rgba(29, 27, 25, 0.06);
      }}
      h1, h2 {{
        margin: 0;
      }}
      p {{
        margin: 0;
        color: var(--muted);
      }}
      .metric {{
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--accent);
      }}
      table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 14px;
        font-size: 0.95rem;
      }}
      th, td {{
        text-align: left;
        padding: 10px 8px;
        border-top: 1px solid var(--line);
        vertical-align: top;
      }}
      th {{
        color: var(--muted);
        font-weight: 600;
      }}
      code {{
        font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
        font-size: 0.85rem;
      }}
    </style>
  </head>
  <body>
    <main>
      <section class="hero">
        <h1>Tensor Training Core Dashboard</h1>
        <p>Lightweight browsing view for recent jobs and registered exported models.</p>
      </section>
      <section class="summary">
        <div class="card">
          <p>Total jobs</p>
          <div class="metric">{escape(str(payload['job_count']))}</div>
        </div>
        <div class="card">
          <p>Registered models</p>
          <div class="metric">{escape(str(payload['model_count']))}</div>
        </div>
        <div class="card">
          <p>Registry index</p>
          <div><code>{escape(str(payload['model_registry_index_path']))}</code></div>
        </div>
      </section>
      <section class="card">
        <h2>Recent Jobs</h2>
        <table>
          <thead>
            <tr>
              <th>Job ID</th>
              <th>Operation</th>
              <th>State</th>
              <th>Updated</th>
              <th>Config</th>
            </tr>
          </thead>
          <tbody>
            {job_rows}
          </tbody>
        </table>
      </section>
      <section class="card" style="margin-top: 20px;">
        <h2>Model Registry</h2>
        <table>
          <thead>
            <tr>
              <th>Model Key</th>
              <th>Latest Version</th>
              <th>Versions</th>
              <th>Descriptor</th>
            </tr>
          </thead>
          <tbody>
            {model_rows}
          </tbody>
        </table>
      </section>
    </main>
  </body>
</html>"""
    return HTMLResponse(html)
