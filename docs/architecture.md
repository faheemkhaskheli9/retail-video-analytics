# Architecture Notes: Retail Video Analytics Platform

## Pipeline

```text
Camera Feed -> Detection (YOLO) -> Tracking -> Zone/Event Logic -> Database -> Analytics Dashboard
```

## Components

- YOLO-based object/person detection
- Multi-object tracking
- Customer counting
- Entry/exit counting
- Dwell-time measurement
- Zone definitions
- Customer heatmaps
- Cashier/staff detection
- Counter-presence detection
- Cashier-absence time tracking
- Historical analytics dashboard

## Design Notes

- Keep provider/model choices swappable behind interfaces (see `multi-llm-router`
  and similar projects in this portfolio for the general pattern).
- Prefer configuration-driven pipelines (YAML/JSON in `configs/`) over hardcoded
  parameters so experiments are reproducible.
