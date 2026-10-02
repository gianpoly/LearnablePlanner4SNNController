import time
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor

from RegressorHelpers import RegressorNet, MotionSequenceDataset
from RegressorPlannerTrainingTesting import simulate_SNN_states, fit_regressor
from optimal_search_utils import read_controller_specs
from closed_loop import run_closed_loop

# DAgger: retrain the planner on the states the SNN visits when the planner itself drives it in closed loop, labelled
# with the GT targets of the same frame, so that it learns to recover from its own errors

motion_sequence_file_path = "./data/walk.amc"
specs_path = "./data/optimal_controller_specs.txt"
hidden_neurons = 128
n_iters = 6  # number of collect + refit rounds
n_rollouts = 4  # closed-loop rollouts per round (run in parallel, each with its own random SNN pool weights)
event_driven = True  # roll out in the mode the planner is deployed in


def rollout(dataset: MotionSequenceDataset, net: RegressorNet, specs_list: list, seed: int) -> np.ndarray:
    np.random.seed(seed)  # SNN motor neuron pool weights
    torch.set_num_threads(1)  # one process per rollout already uses all cores
    frame_seq, _, _ = run_closed_loop(dataset, net, specs_list, event_driven=event_driven, verbose=False)
    return frame_seq


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    dataset = MotionSequenceDataset(motion_sequence_file_path, oversampling=0)
    DOF_no = dataset.Y.shape[1]
    gt_angles = dataset.norm_fact*dataset.motion_seq
    specs_list = read_controller_specs(specs_path)

    # Initial data: SNN states of an open-loop rollout on the GT targets (same as the _snnstate model)
    X_agg = [simulate_SNN_states(dataset, specs_path)]
    Y_agg = [dataset.Y]
    net, _ = fit_regressor(X_agg[0], dataset.Y, hidden_neurons)

    log = []  # iteration, mean closed-loop angle RMSE (deg), training set size
    best_rmse, best_state = np.inf, None
    start = time.time()
    with ProcessPoolExecutor(n_rollouts) as pool:
        for it in range(n_iters + 1):
            # The rollouts of the current planner both evaluate it and provide its successor's new training data
            seeds = [1000*(it + 1) + k for k in range(n_rollouts)]
            frame_seqs = list(pool.map(rollout, [dataset]*n_rollouts, [net]*n_rollouts, [specs_list]*n_rollouts, seeds))
            rmses = [np.sqrt(np.mean((f[:, :DOF_no] - gt_angles)**2)) for f in frame_seqs]
            n_samples = sum(x.shape[0] for x in X_agg)
            log.append([it, np.mean(rmses), n_samples])
            print(f"Iteration {it}: closed-loop angle RMSE {' / '.join(f'{r:.2f}' for r in rmses)} deg "
                  f"(trained on {n_samples} samples, {time.time() - start:.0f} s)")

            if np.mean(rmses) < best_rmse:
                best_rmse, best_state = np.mean(rmses), {k: v.clone() for k, v in net.state_dict().items()}
            if it == n_iters:
                break

            # Aggregate the visited states (normalized like dataset.X: angles/norm_fact, speeds already normalized)
            for f in frame_seqs:
                X_agg.append(np.concatenate((f[:, :DOF_no]/dataset.norm_fact, f[:, DOF_no:]), axis=1))
                Y_agg.append(dataset.Y)
            net, _ = fit_regressor(np.concatenate(X_agg), np.concatenate(Y_agg), hidden_neurons)

    print(f"Best closed-loop angle RMSE {best_rmse:.2f} deg")
    torch.save(best_state, f"./models/planner/Regressor_hid{hidden_neurons}_ov0_dagger.pt")
    np.savetxt("./data/dagger_log.txt", np.array(log), delimiter=",", header="iteration,angle_rmse_deg,n_samples")
