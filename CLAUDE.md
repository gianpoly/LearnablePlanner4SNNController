# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase that reproduces CMU motion-capture sequences (`.amc`/`.asf`) with a spiking-neural-network (SNN) joint controller driven by a learned **planner**. The planner (`RegressorNet`, a 3-layer MLP) maps the character's normalized state (joint angles + angular velocities) to the target angles; per-joint SNN controllers then track those targets.

## Running things

No packaging, requirements file, test suite, or linter. Scripts are standalone and have hardcoded paths. Install dependencies with `uv pip install -r requirements.txt` (see README). The requirement is `pygame-ce`, not `pygame`, because upstream pygame has no Python 3.14 wheel and fails to build without SDL2. The code still does `import pygame`.

**Run every script from the repo root**, e.g. `python src/real_time_plan_SNN_recontr.py`:
- Data paths are CWD-relative (`./data/...`, `./models/planner/...`, `./figs/...`).
- `IO/` gets imported via `sys.path.append('./IO')`, which also only works from the root.
- Modules in `src/` import each other as top-level modules (`from controllers import ...`). That works because Python puts the script's directory (`src/`) on `sys.path`.

`data/`, `models/`, `figs/` and all `*.txt`/`*.amc`/`*.asf`/`*.png`/`*.svg` files are gitignored. Inputs such as `data/walk.amc` and `data/skeleton.asf`, plus trained models, must already be present locally.

## Pipeline (run in this order)

1. **Optimal SNN controller search:** `src/optimal_SNN_controller_search.py`. For each moving joint it runs a heap-based search (`search_ctrlrs_smart`) over controller configurations and writes `data/optimal_controller_specs.txt`. Each row is `joint_id, speed_ctrl, facil×2, PSI×2, MN_pool_size`. Joints that aren't listed fall back to `[0,0,0,0,0,3]`. It needs `data/training_sequence.txt`, which step 2 produces as a side effect.
2. **Train planner:** `train_regressor(...)` in `src/RegressorPlannerTrainingTesting.py` (commented out in `__main__`). It saves `models/planner/Regressor_hid{N}_ov{K}.pt`.
3. **Offline reconstruction:** `src/offline_plan_SNN_reconstr.py`. It queries the planner at a fixed rate on **ground-truth** states, runs each joint's controller independently via `perform_gesture`, and writes `data/walk_reconstr.amc`.
4. **Real-time (closed-loop) reconstruction:** `src/real_time_plan_SNN_recontr.py`. One `AgentController` for all DOFs. Every frame, the SNN's own state is fed back into the planner. It writes `data/reconstructed_{targets,angles,speeds}.txt`. This is the active work area: recent commits note it is slow and that the speed feedback doesn't match (currently the speed columns are copied from ground truth, not from the SNN).
5. **Inspect:** `src/vis_queries.py` plots the real-time outputs. `IO/3Dviewer.py` (pygame/OpenGL) plays back `data/walk_reconstr.amc` on `data/skeleton.asf`.

## Key conventions and gotchas

- **Model filename encodes hyperparameters.** `setup_regressor_testing_env` parses `model_path.split('_')`: `hid` digits give the hidden size, and the single character after `ov` gives oversampling. An underscore anywhere else in the path, or oversampling ≥10, breaks this.
- **Normalization:** `MotionSequenceDataset` divides angles by one global `norm_fact` (the max |angle|) and derivatives by `derivs_norm_fact`. Planner inputs must be normalized with these and outputs multiplied back by `norm_fact`. The dataset object carries both factors, which is why test scripts rebuild it from the `.amc`.
- **Dataset construction has side effects:** it rewrites `data/walk.txt` and `data/training_sequence.txt` every time.
- **Targets are piecewise-constant:** trigger points are the zero-crossings of a 10-frame finite-difference derivative (`dataset_utils.py`), plus the first and last frames, optionally oversampled by midpoints. Joint 2 (root translation) is always oversampled at least 5×. Between triggers the target equals the angle at the next trigger.
- **Time bases:** mocap runs at 120 fps. SNN controllers integrate with `dt = 0.1` ms. Offline scripts cubic/linear-interpolate frames to 1 ms steps (`frames2time`, `generate_SNN_training_set`).
- **DOF layout:** 62 DOFs in CMU order. The hardcoded slice map in `IO/amc_processing.txt2amc` is the reference (e.g. index 51 = `rtibia`, the joint most plots inspect).
- There are two derivative/zero-crossing implementations: `src/dataset_utils.py` (planner, frames × DOF) and `src/optimal_search_utils.py` (SNN search, DOF × time). The axis orders differ.
- SNN model (`src/controllers.py`): LIF `Neuron` / `MotorNeuronPool` → `AdaptiveSingleDOFController` (optional speed control, facilitation, PSI modulation, variable motor-neuron pool size) → `AgentController` (one controller per DOF). Current conversions and weight clipping live in `controllers_utils.py`.
