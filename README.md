# Sentinel-1 LULC project: stage 1

This is the first runnable stage, not the completed coursework submission.
It checks your environment and implements the required percentage-point maths.
It does not yet download Sentinel-1 data, train a model, or display real maps.

## Architecture

The final pipeline will retrieve Sentinel-1 imagery, prepare monthly composites,
apply a trained pixel-classification model, then calculate and display results.
Training labels are reference targets, not additional model input channels.
The statistics module expects aligned maps on a grid with equal ground area per
pixel. It uses only pixels valid in all twelve months. Real geospatial processing
must enforce these conditions before calling it.

## Run in Data Lab

1. Upload and extract this folder into your persistent project storage.
2. Open a terminal in the extracted folder.
3. Run the following commands. If your platform uses `python` instead of
   `python3`, replace that command name.

```sh
python3 check_environment.py
python3 -m unittest -v
python3 run_demo.py
```

In a Python notebook, the equivalent commands are:

```python
%run check_environment.py
%run test_statistics.py
%run run_demo.py
```

All three tests should pass. The demo writes
`results/SYNTHETIC_monthly_statistics.csv`. Built-up increases from 20% to 31%,
which is +11 percentage points. The first month's monthly change is blank,
because no preceding month exists in the selected window.

The three demo classes are placeholders for testing calculations, not an
approved class scheme. The synthetic output is not scientific evidence and
must not be submitted as trained-model results.

## Next decisions

Identify the exact Data Lab platform, GPU availability, persistent storage,
internet access, deadline, study area, geographic extent, dates, real label
source, and independent comparison map. Then implement the real data pipeline,
training, held-out evaluation, browser prototype, and report evidence.

Keep imagery and weights outside Git; provide accessible submission links when
needed. Never commit private API keys. Retain notebooks with executed outputs.
