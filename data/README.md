# Data

## `raw/CMAPSSData/` - NASA C-MAPSS turbofan degradation data

Source: [NASA Prognostics Data Repository](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
(Saxena, Goebel, Simon & Eklund, *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM 2008).
The paper and NASA's own readme are included in that folder.

| File | Content |
|---|---|
| `train_FD00x.txt` | run-to-failure trajectories: the last row of each engine is the failure |
| `test_FD00x.txt` | trajectories cut off at a random point before failure |
| `RUL_FD00x.txt` | true remaining cycles at the last row of each test engine |

Each row has 26 space-separated columns: `engine_id, cycle, setting_1..3, sensor_1..21`.
The sensor meanings (T24, T30, Ps30, ...) are listed in `src/fleetguard/data/loader.py` (`SENSOR_INFO`).

## `incoming/` - new data for retraining (created at runtime)

To have FleetGuard retrain on new run-to-failure data, drop files named `<SUBSET>_<anything>.csv`
(with the columns above as headers) or `.txt` (NASA format) here, then run
`python -m fleetguard retrain --subset <SUBSET>`. Processed files move to `incoming/archive/`.
