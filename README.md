# RegressionPlanner

## Installation

Requires Python 3.10+.

With [uv](https://docs.astral.sh/uv/):

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or with plain pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

```
                 data/walk.amc  (CMU mocap, 120 fps, 62 DOFs)
                              |
                              v
        +-------------------------------------------------+
        | 0. IO/amc_processing.py                         |
        |    .amc -> frames x DOF matrix                  |
        |    (built inside MotionSequenceDataset)         |
        +-------------------------------------------------+
                              |  data/training_sequence.txt
               +--------------+---------------+
               |                              |
               v                              v
+------------------------------+  +------------------------------+
| 1. SNN controller search     |  | 2. Planner training          |
| optimal_SNN_controller_      |  | RegressorPlannerTraining     |
|   search.py                  |  |   Testing.train_regressor    |
|                              |  |                              |
| per joint: sample step-wise  |  | X = [angles, velocities]     |
| targets, best-first search   |  |     (normalized)             |
| over [speed, facil+/-,       |  | Y = target angle (value at   |
| PSI+/-, MN pool size],       |  |     next trigger)            |
| scored by normalized RMSE    |  | MLP 124 -> h -> h -> 62      |
+------------------------------+  +------------------------------+
               |                              |
               | data/optimal_controller_     | models/planner/
               |   specs.txt                  |   Regressor_hid{h}_ov{k}.pt
               +--------------+---------------+
                              |
             +----------------+-----------------+
             |                                  |
             v                                  v
+-----------------------------+  +---------------------------------+
| 3. Offline (open loop)      |  | 4. Real time (closed loop)      |
| offline_plan_SNN_reconstr   |  | real_time_plan_SNN_recontr.py   |
|                             |  |                                 |
| planner queried on GROUND-  |  |  +-> planner(SNN state)         |
| TRUTH states @ 7.5 Hz,      |  |  |        |  targets (per frame) |
| each joint simulated        |  |  |        v                      |
| independently               |  |  +-- AgentController (62 SNNs,  |
| (perform_gesture)           |  |       dt = 0.1 ms)              |
+-----------------------------+  +---------------------------------+
             |                                  |
             | data/walk_reconstr.amc           | data/reconstructed_
             v                                  |   {targets,angles,speeds}.txt
+-----------------------------+                 v
| 5a. IO/3Dviewer.py          |  +---------------------------------+
|     skeleton playback       |  | 5b. vis_queries.py              |
+-----------------------------+  |     plot vs ground truth        |
                                 +---------------------------------+
```

Note: step 1 reads `data/training_sequence.txt`, which is written when the planner dataset is built (step 2), so build the dataset at least once first.

Run all scripts from the repository root (e.g. `python src/real_time_plan_SNN_recontr.py`), since data, model and figure paths are relative to it. The `data/`, `models/` and `figs/` directories are not tracked and must be created/populated locally.
