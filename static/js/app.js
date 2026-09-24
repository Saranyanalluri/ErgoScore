/**
 * ErgoScore front-end charts.
 *
 * Reads the analysis result out of the #result-data JSON script tag (put
 * there by results.html via Jinja) and renders it with Chart.js. This file
 * does not know or care whether the underlying data is real or dummy — it
 * only depends on the shape produced by app/pipeline.py.
 */

document.addEventListener("DOMContentLoaded", () => {
  const dataEl = document.getElementById("result-data");
  if (!dataEl) {
    // Not on the results page — nothing to do.
    return;
  }

  let data;
  try {
    data = JSON.parse(dataEl.textContent);
  } catch (err) {
    console.error("Could not parse result data for charts:", err);
    return;
  }

  renderRiskDistributionChart(data.risk_distribution);
  renderScoreTimelineChart(data.timeline);
});

function renderRiskDistributionChart(distribution) {
  const canvas = document.getElementById("risk-distribution-chart");
  if (!canvas || !distribution) return;

  new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: ["Low", "Medium", "High"],
      datasets: [
        {
          data: [distribution.low, distribution.medium, distribution.high],
          backgroundColor: ["#1f7a4d", "#b06a00", "#a4222c"],
          borderColor: "#ffffff",
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: "bottom",
          labels: { boxWidth: 12, font: { size: 12 } },
        },
        tooltip: {
          callbacks: {
            label: (context) => `${context.label}: ${context.parsed}%`,
          },
        },
      },
    },
  });
}

function renderScoreTimelineChart(timeline) {
  const canvas = document.getElementById("score-timeline-chart");
  if (!canvas || !timeline) return;

  const labels = timeline.map((point) => `${point.time}s`);
  const scores = timeline.map((point) => point.score);

  new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "REBA Score",
          data: scores,
          borderColor: "#1e2761",
          backgroundColor: "rgba(30, 39, 97, 0.08)",
          fill: true,
          tension: 0.3,
          pointRadius: 3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          suggestedMax: 15,
          title: { display: true, text: "REBA Score" },
        },
        x: {
          title: { display: true, text: "Time" },
        },
      },
      plugins: {
        legend: { display: false },
      },
    },
  });
}
