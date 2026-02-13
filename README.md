# Ambulance Vitals — Anomaly Detection Pipeline

[![Publish Docker image](https://github.com/keanesc/time-series-analysis/actions/workflows/publish-ghcr.yml/badge.svg)](https://github.com/keanesc/time-series-analysis/actions)
[![GHCR image](https://img.shields.io/badge/ghcr.io%2Fkeanesc%2Ftime--series--analysis-latest-blue?logo=github&logoColor=white)](https://github.com/keanesc?tab=packages)

Production-oriented pipeline for early-warning anomaly detection and triage risk scoring from streamed vitals (HR, SpO2, ABP).

## Quick start

### Default (recommended): run the published GHCR image

```bash
docker pull ghcr.io/keanesc/time-series-analysis:latest
docker run -p 8000:8000 ghcr.io/keanesc/time-series-analysis:latest
```

Verify the service:

```bash
curl http://localhost:8000/health
```

### Manual installation (from source)

This project uses [pixi](https://pixi.sh) for environment management. Use the commands below when developing locally or running from source.

```bash
# clone the repo (SSH or HTTPS)
cd time-series-analysis
pixi install
pixi run serve  # open http://localhost:8000
pixi run all    # run full pipeline
```

## Core commands

- `pixi run all` — run entire pipeline end-to-end
- `pixi run collect` — fetch & simulate motion
- `pixi run clean` — artifact detection + cleaning
- `pixi run train` — train anomaly model
- `pixi run score` — compute risk scores
- `pixi run evaluate` — metrics & latency
- `pixi run failures` — failure-case analysis
- `pixi run serve` — start API server

## API (essentials)

- `GET /health` — liveness
- `POST /predict` — input: `hr`, `spo2`, `sbp`, `dbp`, `motion`
  returns: `anomaly_flag`, `risk_score`, `confidence`, `alert_status`

Example:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"hr":120,"spo2":88,"sbp":85,"dbp":55,"motion":0.1}'
```

Response:

```json
{
  "anomaly_flag": true,
  "risk_score": 78.3,
  "confidence": 0.87,
  "alert_status": "ALERT"
}
```

## Docker

Build locally:

```bash
docker build -t time-series-analysis .
docker run -p 8000:8000 time-series-analysis
```

Published image (GHCR):

```bash
docker pull ghcr.io/keanesc/time-series-analysis:latest
docker run -p 8000:8000 ghcr.io/keanesc/time-series-analysis:latest
```

The repository includes a GitHub Actions workflow (`.github/workflows/publish-ghcr.yml`) that builds and publishes the image to GHCR on pushes to `main`. (Image is public.)

## Data Source

**MIMIC-III Waveform Database** (PhysioNet, v1.0) [1]

- 67,830 record sets from ~30,000 ICU patients
- Numerics records at 1 Hz: HR, SpO₂, ABP (systolic/diastolic)
- Open access under ODbL v1.0
- Motion/vibration signal is **simulated** (ambulance context adaptation)

## Model Approach

1. **Clinical features** (per 60 s window): distance-to-threshold, slope, mean, std,
   end value for each vital, plus motion and data quality indicators (24 features total)
2. **HistGradientBoostingClassifier** trained on stable windows to predict pre-critical
   deterioration within 2 minutes
3. **Risk score** (0–100): weighted vital severities + trend bonus, gated by
   data-quality confidence
4. **Alert logic**: ALERT if risk ≥ 70 and confidence ≥ 0.6; SUPPRESSED if
   confidence too low

See [report.md](report.md) for detailed methodology, metrics, and safety-critical analysis.

## References

[1] PhysioNet, “MIMIC‑III Waveform Database (v1.0).” [Online]. Available: [physionet.org/content/mimic3wdb/1.0](https://physionet.org/content/mimic3wdb/1.0/)

[2] P. Zwerschke, “Shipping conda environments to production using pixi,” QuantCo Tech Blog, Jul. 11, 2024. [Online]. Available: [tech.quanto.com/blog/pixi-production](https://tech.quantco.com/blog/pixi-production)
