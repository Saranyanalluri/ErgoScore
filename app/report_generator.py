"""
report_generator.py
====================

Person 3 — Reporting & visualization module for ErgoScore.

Takes the analysis dict produced by risk_analyzer.create_analysis() (or
loaded back from analysis.json) and produces:

    * reports/score_timeline.png      — REBA score over time
    * reports/risk_distribution.png   — low/medium/high time breakdown
    * reports/report.html             — a simple, self-contained HTML
                                         report combining everything

This module does NOT recalculate REBA scores or perform any temporal
analysis itself — it only visualizes and summarizes results that were
already computed by risk_analyzer.py.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, Union

import matplotlib

matplotlib.use("Agg")  # non-interactive backend, safe for headless/server use
import matplotlib.pyplot as plt

PathLike = Union[str, Path]

# Colors used consistently across both graphs and the HTML report.
RISK_COLORS = {
    "low": "#2e7d32",     # green
    "medium": "#f9a825",  # amber
    "high": "#c62828",    # red
}


# ---------------------------------------------------------------------------
# A. SCORE TIMELINE GRAPH
# ---------------------------------------------------------------------------
def plot_score_timeline(analysis: Dict[str, Any], output_path: PathLike) -> Path:
    """Create a line graph of REBA score vs. time (raw + smoothed).

    Args:
        analysis: The analysis dict (must contain a "frames" list with
            "timestamp", "raw_score", "smoothed_score" per entry).
        output_path: Where to save the PNG.

    Returns:
        The resolved Path of the saved image.
    """
    frames = analysis["frames"]
    timestamps = [f["timestamp"] for f in frames]
    raw_scores = [f["raw_score"] for f in frames]
    smoothed_scores = [f["smoothed_score"] for f in frames]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        timestamps, raw_scores,
        color="#90a4ae", linewidth=1, alpha=0.6, label="Raw REBA score",
    )
    ax.plot(
        timestamps, smoothed_scores,
        color="#1565c0", linewidth=2.2,
        label=f"Smoothed (window={analysis.get('smoothing_window', '?')})",
    )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("REBA score")
    ax.set_title("REBA Score Over Time")
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# B. RISK DISTRIBUTION GRAPH
# ---------------------------------------------------------------------------
def plot_risk_distribution(analysis: Dict[str, Any], output_path: PathLike) -> Path:
    """Create a bar chart of the percentage of time in each risk category.

    Args:
        analysis: The analysis dict (must contain "risk_distribution").
        output_path: Where to save the PNG.

    Returns:
        The resolved Path of the saved image.
    """
    distribution = analysis["risk_distribution"]
    categories = list(distribution.keys())
    values = [distribution[c] for c in categories]
    colors = [RISK_COLORS.get(c, "#78909c") for c in categories]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(
        [c.capitalize() for c in categories], values, color=colors, width=0.6
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_ylabel("Percentage of frames (%)")
    ax.set_title("Risk Distribution")
    ax.set_ylim(0, max(100, max(values) + 10) if values else 100)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# C. PLAIN-LANGUAGE SUMMARY
# ---------------------------------------------------------------------------
def generate_summary(analysis: Dict[str, Any]) -> str:
    """Generate a short, factual, plain-language summary of the analysis.

    Only states things directly supported by the calculated results.
    Does not make medical claims or predict injury.

    Args:
        analysis: The analysis dict.

    Returns:
        A summary string, 2-4 sentences.
    """
    duration = analysis["video_duration"]
    avg = analysis["average_score"]
    max_score = analysis["maximum_score"]
    min_score = analysis["minimum_score"]
    distribution = analysis["risk_distribution"]
    worst_frames = analysis["worst_frames"]

    sentences = [
        f"The analyzed activity lasted {duration:.1f} seconds across "
        f"{analysis['frame_count']} frames.",
        f"The average REBA score was {avg:.1f} (range: {min_score:.0f}-{max_score:.0f}).",
    ]

    dominant_category = max(distribution, key=distribution.get)
    sentences.append(
        f"Postures were most frequently classified as {dominant_category} risk "
        f"({distribution[dominant_category]:.1f}% of frames), with "
        f"{distribution.get('low', 0):.1f}% low, "
        f"{distribution.get('medium', 0):.1f}% medium, and "
        f"{distribution.get('high', 0):.1f}% high risk."
    )

    if worst_frames:
        worst = worst_frames[0]
        sentences.append(
            f"The highest observed REBA score was {worst['score']:.0f}, "
            f"occurring at approximately {worst['timestamp']:.1f} seconds "
            f"into the clip (frame {worst['frame']})."
        )

    return " ".join(sentences)


# ---------------------------------------------------------------------------
# D. HTML REPORT
# ---------------------------------------------------------------------------
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ErgoScore Report</title>
<style>
  body {{
    font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
    max-width: 900px;
    margin: 40px auto;
    padding: 0 20px;
    color: #212121;
    background: #fafafa;
  }}
  h1 {{ color: #1565c0; margin-bottom: 4px; }}
  .subtitle {{ color: #757575; margin-top: 0; margin-bottom: 30px; }}
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin-bottom: 30px;
  }}
  .stat-card {{
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 14px 16px;
  }}
  .stat-card .label {{ font-size: 12px; color: #757575; text-transform: uppercase; letter-spacing: 0.03em; }}
  .stat-card .value {{ font-size: 22px; font-weight: 600; margin-top: 4px; }}
  section {{ margin-bottom: 36px; }}
  h2 {{ border-bottom: 2px solid #e0e0e0; padding-bottom: 6px; color: #333; }}
  img {{ max-width: 100%; border: 1px solid #e0e0e0; border-radius: 6px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
  th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid #e0e0e0; }}
  th {{ background: #f0f0f0; }}
  .summary-box {{
    background: white;
    border-left: 4px solid #1565c0;
    padding: 14px 18px;
    border-radius: 4px;
    line-height: 1.5;
  }}
  .risk-badge {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    color: white;
    font-size: 12px;
    font-weight: 600;
  }}
  footer {{ color: #9e9e9e; font-size: 12px; margin-top: 50px; text-align: center; }}
</style>
</head>
<body>
  <h1>ErgoScore</h1>
  <p class="subtitle">Ergonomic Posture Risk Assessment &mdash; Temporal Analysis Report</p>

  <section>
    <div class="stats-grid">
      <div class="stat-card"><div class="label">Video Duration</div><div class="value">{duration:.1f}s</div></div>
      <div class="stat-card"><div class="label">Frames Analyzed</div><div class="value">{frame_count}</div></div>
      <div class="stat-card"><div class="label">Average Score</div><div class="value">{average_score:.1f}</div></div>
      <div class="stat-card"><div class="label">Minimum Score</div><div class="value">{minimum_score:.0f}</div></div>
      <div class="stat-card"><div class="label">Maximum Score</div><div class="value">{maximum_score:.0f}</div></div>
    </div>
  </section>

  <section>
    <h2>Summary</h2>
    <div class="summary-box">{summary}</div>
  </section>

  <section>
    <h2>Score Timeline</h2>
    <img src="{timeline_img}" alt="REBA score over time">
  </section>

  <section>
    <h2>Risk Distribution</h2>
    <img src="{distribution_img}" alt="Risk distribution chart">
    <table>
      <tr><th>Category</th><th>Percentage of frames</th></tr>
      {risk_rows}
    </table>
  </section>

  <section>
    <h2>Worst Frames</h2>
    <table>
      <tr><th>Frame</th><th>Timestamp (s)</th><th>REBA Score</th></tr>
      {worst_frame_rows}
    </table>
  </section>

  <footer>
    Generated by ErgoScore &mdash; Person 3 (Temporal Risk Analysis) module.
    This report is an ergonomic screening aid, not a medical diagnosis.
  </footer>
</body>
</html>
"""


def generate_html_report(
    analysis: Dict[str, Any],
    summary: str,
    timeline_img_path: PathLike,
    distribution_img_path: PathLike,
    output_path: PathLike,
) -> Path:
    """Render the final HTML report to disk.

    Args:
        analysis: The analysis dict.
        summary: The plain-language summary string (see generate_summary()).
        timeline_img_path: Path to the score-timeline PNG (used for the
            <img src> — stored as a relative filename so the report works
            when opened locally next to the images).
        distribution_img_path: Path to the risk-distribution PNG.
        output_path: Where to write report.html.

    Returns:
        The resolved Path of the written HTML file.
    """
    risk_rows = "\n".join(
        f'<tr><td><span class="risk-badge" style="background:{RISK_COLORS.get(cat, "#78909c")}">'
        f'{html.escape(cat.capitalize())}</span></td><td>{pct:.1f}%</td></tr>'
        for cat, pct in analysis["risk_distribution"].items()
    )

    worst_frame_rows = "\n".join(
        f"<tr><td>{wf['frame']}</td><td>{wf['timestamp']:.2f}</td><td>{wf['score']:.0f}</td></tr>"
        for wf in analysis["worst_frames"]
    )
    if not worst_frame_rows:
        worst_frame_rows = '<tr><td colspan="3">No frame data available.</td></tr>'

    rendered = _HTML_TEMPLATE.format(
        duration=analysis["video_duration"],
        frame_count=analysis["frame_count"],
        average_score=analysis["average_score"],
        minimum_score=analysis["minimum_score"],
        maximum_score=analysis["maximum_score"],
        summary=html.escape(summary),
        timeline_img=Path(timeline_img_path).name,
        distribution_img=Path(distribution_img_path).name,
        risk_rows=risk_rows,
        worst_frame_rows=worst_frame_rows,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# ORCHESTRATION — used internally by risk_analyzer.analyze_scores()
# ---------------------------------------------------------------------------
def generate_reports(analysis: Dict[str, Any], output_directory: PathLike) -> Dict[str, str]:
    """Generate both graphs, the summary, and the HTML report in one call.

    Args:
        analysis: The analysis dict produced by risk_analyzer.create_analysis().
        output_directory: Directory to write score_timeline.png,
            risk_distribution.png, and report.html into.

    Returns:
        Dict with keys "score_timeline_png", "risk_distribution_png",
        and "report_html", mapping to the string paths of each file.
    """
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    timeline_path = plot_score_timeline(analysis, output_dir / "score_timeline.png")
    distribution_path = plot_risk_distribution(analysis, output_dir / "risk_distribution.png")
    summary = generate_summary(analysis)
    report_path = generate_html_report(
        analysis, summary, timeline_path, distribution_path, output_dir / "report.html"
    )

    return {
        "score_timeline_png": str(timeline_path),
        "risk_distribution_png": str(distribution_path),
        "report_html": str(report_path),
        "summary": summary,
    }
