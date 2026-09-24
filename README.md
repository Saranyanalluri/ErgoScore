# ErgoScore — Person 4 Layer (Flask App, Dashboard, Integration Pipeline)

This is the **Person 4** slice of the ErgoScore project: the Flask backend,
upload workflow, results dashboard, and the single integration point
(`app/pipeline.py`) the other three team members' modules will plug into.

**Everything in this app currently runs on dummy data.** No real video
analysis, pose detection, or REBA scoring happens yet — see
[Integration point](#integration-point-for-person-1-2-3) below.

## Project structure

```
ErgoScore/
├── app/
│   ├── __init__.py      # Flask app factory, config, error handlers
│   ├── pipeline.py       # analyze_video() — INTEGRATION POINT (dummy for now)
│   └── routes.py         # GET /, POST /analyze, video-serving route
├── static/
│   ├── css/style.css
│   └── js/app.js         # Chart.js rendering for the dashboard
├── templates/
│   ├── index.html        # Upload page
│   ├── results.html       # Results dashboard
│   └── error.html
├── videos/
│   ├── uploads/           # Uploaded videos are saved here
│   └── processed/         # Reserved for processed/annotated video output
├── data/                  # Reserved for intermediate analysis data
├── snapshots/              # Reserved for extracted frame images
├── reports/                # Reserved for generated report files
├── tests/
│   └── test_app.py
├── run.py
├── requirements.txt
└── .gitignore
```

## Setup

```bash
# from inside the ErgoScore/ folder
python -m venv venv

# activate it
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

## Run

```bash
python run.py
```

Then open **http://127.0.0.1:5000** in a browser.

## Testing checklist (manual, dummy-data workflow)

1. Start the app (`python run.py`) and open `http://127.0.0.1:5000`.
2. Confirm the upload page loads with a video file input, a **Load**
   dropdown, a **Coupling** dropdown, and an **Analyze Video** button.
3. Click **Analyze Video** with no file selected → you should be redirected
   back to the upload page with an error message ("Please choose a video
   file to upload.").
4. Upload a non-video file (e.g. a `.txt` renamed, or any unsupported
   extension) → you should see an "Unsupported file type" error.
5. Upload a real short video file (any `.mp4`/`.mov`/`.avi`/`.webm`/`.mkv`)
   with **Load = Heavy** and **Coupling = Poor**, then click **Analyze
   Video**.
6. On the results page, confirm you see:
   - A notice banner stating this is placeholder analysis.
   - The uploaded video, playable inline.
   - An overall REBA score and a risk badge (Low/Medium/High).
   - Four key metrics: average score, maximum score, duration, high-risk
     exposure percentage.
   - A risk distribution doughnut chart (Low/Medium/High %).
   - A score timeline line chart.
   - 3–5 "worst posture frame" cards with frame number, timestamp, score,
     and trunk/neck/knee angles (image shows "No snapshot available").
   - A plain-language summary paragraph.
7. Click **Analyze another video** and confirm it returns you to the
   upload page.
8. Re-run the analysis with different **Load**/**Coupling** values and
   confirm the scores change (the dummy pipeline nudges scores based on
   these inputs, so this is a visible sanity check that the values are
   actually reaching `analyze_video()`).

## Automated tests

```bash
pytest
```

This covers: the upload page loading, missing-file handling, invalid file
type handling, a full analyze round-trip rendering the dashboard, the
uploaded file actually being saved, and the shape (keys/types) of the dict
`analyze_video()` returns.

## Integration point for Person 1/2/3

All real analysis work plugs into **`app/pipeline.py`**, specifically the
`analyze_video(video_path, load, coupling)` function. It currently returns
placeholder data, but the **shape of the returned dictionary is the
contract** the rest of the app (routes, templates, charts) is built
against:

```python
{
    "overall_score": int,
    "risk_level": "Low" | "Medium" | "High",
    "average_score": float,
    "maximum_score": int,
    "duration": str,
    "risk_distribution": {"low": float, "medium": float, "high": float},
    "timeline": [{"time": int, "score": int}, ...],
    "worst_frames": [
        {
            "frame_number": int,
            "timestamp": str,
            "score": int,
            "trunk_angle": float,
            "neck_angle": float,
            "knee_angle": float,
            "image_url": str | None,
        },
        ...
    ],
    "summary": str,
}
```

Planned pipeline once all modules are ready:

```
video_path
  → Person 1: MediaPipe pose extraction (keypoints per frame)
  → Person 2: joint angles + REBA/RULA scoring (per-frame score)
  → Person 3: temporal risk analysis + report generation
  → analyze_video() returns the dict above, built from real data
```

As long as the returned dict keeps this shape, **no changes should be
needed in `routes.py`, the templates, or `app.js`.**

## Notes

- Uploaded videos are stored in `videos/uploads/` and served back to the
  browser via the `/videos/uploads/<filename>` route — they are not placed
  under `static/`, so filenames are sanitized with `secure_filename` and
  checked against directory traversal before being served.
- Max upload size is capped at 200 MB (`MAX_CONTENT_LENGTH` in
  `app/__init__.py`); uploads over that limit get a friendly error instead
  of a raw server error.
- No tracebacks are ever shown to the user — unexpected errors are logged
  server-side and rendered through `templates/error.html`.
