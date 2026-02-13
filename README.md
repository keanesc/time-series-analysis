# Ambulance Vitals — Anomaly Detection Pipeline

Production-oriented pipeline for early-warning anomaly detection and triage risk scoring from streamed vitals (HR, SpO2, ABP).

## Quick start

This project uses [pixi](https://pixi.sh) for environment management [2].

- Install: `pixi install`
- Run API: `pixi run serve` → [local](http://localhost:8000)
- Run full pipeline: `pixi run all`

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
`curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"hr":120,"spo2":88,"sbp":85,"dbp":55,"motion":0.1}'`

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

```bash
docker build -t ambulance-ml .
docker run -p 8000:8000 ambulance-ml
```

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
