from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import sys, os
from modt.utils import (
    compute_hypervolume,
    compute_sparsity,
    check_dominated,
    undominated_indices,
)
from copy import deepcopy
import pickle


def visualize(rollout_logs, logsdir, cur_step, only_hv_sp=False, infos={}):
    n_obj = rollout_logs["n_obj"]
    dataset_min_prefs = rollout_logs["dataset_min_prefs"]
    dataset_max_prefs = rollout_logs["dataset_max_prefs"]
    dataset_min_raw_r = rollout_logs["dataset_min_raw_r"]
    dataset_max_raw_r = rollout_logs["dataset_max_raw_r"]
    dataset_min_final_r = rollout_logs["dataset_min_final_r"]
    dataset_max_final_r = rollout_logs["dataset_max_final_r"]
    target_returns = rollout_logs["target_returns"]
    target_prefs = rollout_logs["target_prefs"]
    rollout_unweighted_raw_r = rollout_logs["rollout_unweighted_raw_r"]
    rollout_weighted_raw_r = rollout_logs["rollout_weighted_raw_r"]
    rollout_original_raw_r = rollout_logs["rollout_original_raw_r"]

    indices_wanted = undominated_indices(rollout_unweighted_raw_r, tolerance=0.05)
    n_points = len(indices_wanted)
    edge_colors = [
        "royalblue" if i in indices_wanted else "r"
        for i in range(rollout_unweighted_raw_r.shape[0])
    ]
    face_colors = ["none" for i in range(rollout_unweighted_raw_r.shape[0])]

    hv = compute_hypervolume(
        rollout_original_raw_r
    )  # this automatically ignores the dominated points
    indices_wanted_strict = undominated_indices(rollout_original_raw_r, tolerance=0)
    front_return_batch = rollout_original_raw_r[indices_wanted_strict]
    sparsity = compute_sparsity(front_return_batch)

    print(f"hv={hv}, sp={sparsity}, n_points={n_points}")

    if not only_hv_sp:
        fig, axes = plt.subplots(n_obj, 3, constrained_layout=True, figsize=(12, 8))
    else:
        fsize = (16, 8) if infos["env"] == "MO-Hopper-v3" else (8, 4)
        fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=fsize)
    axes = axes.flatten()
    fig.add_subplot(111, frameon=False)
    plt.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    sns.despine()
    cur_ax = 0

    # obj0 vs obj1, unweighted
    if n_obj == 2:
        axes[cur_ax].scatter(
            rollout_original_raw_r[:, 0],
            rollout_original_raw_r[:, 1],
            label=f"hv: {hv:.3e}\npts: {n_points}\nsp: {np.round(sparsity, 2)}",
            facecolors=face_colors,
            edgecolors=edge_colors,
        )
        axes[cur_ax].set_xlim([0, max(rollout_original_raw_r[:, 0]) * 1.05])
        axes[cur_ax].set_ylim([0, max(rollout_original_raw_r[:, 1]) * 1.05])
        axes[cur_ax].set_title(f"Obj 0 vs Obj 1")
        axes[cur_ax].set(xlabel="Obj 0", ylabel="Obj 1")
        axes[cur_ax].legend(loc="center left")
        cur_ax += 1
    # change to 3d pareto front
    elif n_obj == 3:
        axes[cur_ax].remove()
        axes[cur_ax] = fig.add_subplot(n_obj, 3, cur_ax + 1, projection="3d")
        axes[cur_ax].scatter(
            rollout_original_raw_r[:, 0],
            rollout_original_raw_r[:, 1],
            rollout_original_raw_r[:, 2],
            label=f"hv: {hv:.3e}\npts: {n_points}\nsp: {np.round(sparsity, 2)}",
            facecolors=face_colors,
            edgecolors=edge_colors,
        )
        axes[cur_ax].set_xlim3d([0, max(rollout_original_raw_r[:, 0]) * 1.05])
        axes[cur_ax].set_ylim3d([0, max(rollout_original_raw_r[:, 1]) * 1.05])
        axes[cur_ax].set_zlim3d([0, max(rollout_original_raw_r[:, 2]) * 1.05])
        axes[cur_ax].set_title(f"Obj 1 vs. Obj 2 vs. Obj 3")
        axes[cur_ax].set(xlabel="Obj 1", ylabel="Obj 2", zlabel="Obj 3")
        axes[cur_ax].legend(loc="lower center")
        cur_ax += 1

    if not only_hv_sp:
        rollout_ratio = rollout_original_raw_r / np.sum(
            rollout_original_raw_r, axis=1, keepdims=True
        )
        axes[cur_ax].scatter(
            target_prefs[:, 0],
            rollout_ratio[:, 0],
            label="MODT",
            facecolors=face_colors,
            edgecolors=edge_colors,
        )
        axes[cur_ax].axvline(
            x=dataset_min_prefs[0],
            ls="--",
        )
        axes[cur_ax].axvline(
            x=dataset_max_prefs[0],
            ls="--",
        )
        axes[cur_ax].set_xlim([-0.05, 1.05])
        axes[cur_ax].set_ylim([-0.05, 1.05])
        axes[cur_ax].set_title(f"Preference 0: Target vs. Achieved")
        axes[cur_ax].set(xlabel="target pref0", ylabel="achieved pref0")
        lims = [
            np.min(
                [axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]
            ),  # min of both axes
            np.max(
                [axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]
            ),  # max of both axes
        ]
        axes[cur_ax].plot(lims, lims, label="oracle", alpha=0.75, zorder=0)
        axes[cur_ax].legend(loc="upper left")
        cur_ax += 1

        axes[cur_ax].scatter(
            target_prefs[:, 1],
            rollout_ratio[:, 1],
            label="MODT",
            facecolors=face_colors,
            edgecolors=edge_colors,
        )
        axes[cur_ax].axvline(
            x=dataset_min_prefs[1],
            ls="--",
        )
        axes[cur_ax].axvline(
            x=dataset_max_prefs[1],
            ls="--",
        )
        axes[cur_ax].set_xlim([-0.05, 1.05])
        axes[cur_ax].set_ylim([-0.05, 1.05])
        axes[cur_ax].set_title(f"Preference 1: Target vs. Achieved")
        axes[cur_ax].set(xlabel="target pref1", ylabel="achieved pref1")
        lims = [
            np.min(
                [axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]
            ),  # min of both axes
            np.max(
                [axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]
            ),  # max of both axes
        ]
        axes[cur_ax].plot(lims, lims, label="oracle", alpha=0.75, zorder=0)
        axes[cur_ax].legend(loc="upper left")
        cur_ax += 1

        # need 1 more graph
        if n_obj == 3:
            axes[cur_ax].scatter(
                target_prefs[:, 2],
                rollout_ratio[:, 2],
                label="MODT",
                facecolors=face_colors,
                edgecolors=edge_colors,
            )
            axes[cur_ax].axvline(
                x=dataset_min_prefs[2],
                ls="--",
            )
            axes[cur_ax].axvline(
                x=dataset_max_prefs[2],
                ls="--",
            )
            axes[cur_ax].set_xlim([-0.05, 1.05])
            axes[cur_ax].set_ylim([-0.05, 1.05])
            axes[cur_ax].set_title(f"Preference 2: Target vs. Achieved")
            axes[cur_ax].set(xlabel="target pref2", ylabel="achieved pref2")
            lims = [
                np.min(
                    [axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]
                ),  # min of both axes
                np.max(
                    [axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]
                ),  # max of both axes
            ]
            axes[cur_ax].plot(lims, lims, label="oracle", alpha=0.75, zorder=0)
            axes[cur_ax].legend(loc="upper left")
            cur_ax += 1

        # rtg0 vs return0, rtg1 vs return1, ... (all are weighted)

        using_mo_rtg = False if len(target_returns.shape) == 1 else True
        if using_mo_rtg:
            axes[cur_ax].scatter(
                target_returns[:, 0],
                rollout_weighted_raw_r[:, 0],
                facecolors=face_colors,
                edgecolors=edge_colors,
                label="MODT",
            )
            axes[cur_ax].set_xlim([-5, np.max(target_returns[:, 0]) * 1.05])
            axes[cur_ax].set_ylim([-5, np.max(rollout_weighted_raw_r[:, 0]) * 1.05])
            axes[cur_ax].set(xlabel="target obj0", ylabel="achieved obj0")
            lims = [
                np.min(
                    [axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]
                ),  # min of both axes
                np.max(
                    [axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]
                ),  # max of both axes
            ]
            axes[cur_ax].plot(lims, lims, label="oracle", alpha=0.75, zorder=0)
            axes[cur_ax].axvline(
                x=dataset_min_raw_r[0],
                ls="--",
            )
            axes[cur_ax].axvline(
                x=dataset_max_raw_r[0],
                ls="--",
            )
            axes[cur_ax].legend(loc="upper left")
            axes[cur_ax].set_title(f"Weighted Obj 0: Target vs. Achieved")
            cur_ax += 1

            axes[cur_ax].scatter(
                target_returns[:, 1],
                rollout_weighted_raw_r[:, 1],
                facecolors=face_colors,
                edgecolors=edge_colors,
                label="MODT",
            )
            axes[cur_ax].set_xlim([-5, np.max(target_returns[:, 1]) * 1.05])
            axes[cur_ax].set_ylim([-5, np.max(rollout_weighted_raw_r[:, 1]) * 1.05])
            axes[cur_ax].set(xlabel="target obj1", ylabel="achieved obj1")
            axes[cur_ax].legend(loc="upper left")
            lims = [
                np.min(
                    [axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]
                ),  # min of both axes
                np.max(
                    [axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]
                ),  # max of both axes
            ]
            axes[cur_ax].plot(lims, lims, label="oracle", alpha=0.75, zorder=0)
            axes[cur_ax].axvline(
                x=dataset_min_raw_r[1],
                ls="--",
            )
            axes[cur_ax].axvline(
                x=dataset_max_raw_r[1],
                ls="--",
            )
            axes[cur_ax].legend(loc="upper left")
            axes[cur_ax].set_title(f"Weighted Obj 1: Target vs. Achieved")
            cur_ax += 1

            if n_obj == 3:
                axes[cur_ax].scatter(
                    target_returns[:, 2],
                    rollout_weighted_raw_r[:, 2],
                    facecolors=face_colors,
                    edgecolors=edge_colors,
                    label="MODT",
                )
                axes[cur_ax].set_xlim([-5, np.max(target_returns[:, 2]) * 1.05])
                axes[cur_ax].set_ylim([-5, np.max(rollout_weighted_raw_r[:, 2]) * 1.05])
                axes[cur_ax].set(xlabel="target obj2", ylabel="achieved obj2")
                axes[cur_ax].legend(loc="upper left")
                lims = [
                    np.min(
                        [axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]
                    ),  # min of both axes
                    np.max(
                        [axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]
                    ),  # max of both axes
                ]
                axes[cur_ax].plot(lims, lims, label="oracle", alpha=0.75, zorder=0)
                axes[cur_ax].axvline(
                    x=dataset_min_raw_r[2],
                    ls="--",
                )
                axes[cur_ax].axvline(
                    x=dataset_max_raw_r[2],
                    ls="--",
                )
                axes[cur_ax].legend(loc="upper left")
                axes[cur_ax].set_title(f"Weighted Obj 2: Target vs. Achieved")
                cur_ax += 1
        else:
            rollout_final_r = np.sum(rollout_weighted_raw_r, axis=1)
            axes[cur_ax].scatter(
                target_returns,
                rollout_final_r,
                facecolors=face_colors,
                edgecolors=edge_colors,
                label="MODT",
            )
            axes[cur_ax].set_xlim([-5, np.max(target_returns) * 1.05])
            axes[cur_ax].set_ylim([-5, np.max(rollout_final_r) * 1.05])
            axes[cur_ax].set(
                xlabel="target final reward", ylabel="achieved final reward"
            )
            lims = [
                np.min(
                    [axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]
                ),  # min of both axes
                np.max(
                    [axes[cur_ax].get_xlim(), axes[cur_ax].get_ylim()]
                ),  # max of both axes
            ]
            axes[cur_ax].plot(lims, lims, label="oracle", alpha=0.75, zorder=0)
            axes[cur_ax].axvline(
                x=dataset_min_final_r,
                ls="--",
            )
            axes[cur_ax].axvline(
                x=dataset_max_final_r,
                ls="--",
            )
            axes[cur_ax].legend(loc="upper left")
            axes[cur_ax].set_title(f"Final Reward: Target vs. Achieved")
            cur_ax += 1

    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    if not os.path.exists(logsdir):
        os.mkdir(logsdir)
    if not only_hv_sp:
        plt.savefig(f"{logsdir}/step={cur_step}_plots.png")
    else:
        env_name = infos["env"]
        dataset = infos["dataset"]
        num_traj = infos["num_traj"]
        plt.savefig(f"{logsdir}/{env_name}_{num_traj}_{dataset}_plots.png")
    plt.close()

    def visualize_pareto_front_all_envs():
        pass


def cal_from_data(
    datasets=["expert_wide"], env_name="MO-Hopper-v2", num_traj=50000, num_plot=1000, data_path="data_collected"
):
    assert num_traj >= num_plot
    generation_path = f"data_generation/{data_path}"
    dataset_paths = [
        f"{generation_path}/{env_name}/{env_name}_{num_traj}_new{d}.pkl"
        for d in datasets
    ]
    trajectories = []
    for data_path in dataset_paths:
        with open(data_path, "rb") as f:
            trajectories.extend(pickle.load(f))

    random_inds = np.random.choice(
        np.arange(len(trajectories)), num_plot, replace=False
    )
    trajectories = np.array(trajectories)[random_inds]

    states, traj_lens, returns, returns_mo, preferences = [], [], [], [], []

    for traj in trajectories:  # just count all trajs
        traj["rewards"] = np.sum(
            np.multiply(traj["raw_rewards"], traj["preference"]), axis=1
        )
        states.append(traj["observations"])
        traj_lens.append(len(traj["observations"]))
        returns.append(traj["rewards"].sum())
        returns_mo.append(traj["raw_rewards"].sum(axis=0))
        preferences.append(traj["preference"][0, :])

    # padding a few (~34/50000) state trajs with 0 to be as long as others.
    max_len = np.max([len(s) for s in states])
    for i, s in enumerate(states):
        if len(s) < max_len:
            states[i] = np.pad(s, ((0, max_len - len(s)), (0, 0)), mode="constant")

    traj_lens, returns, returns_mo, states, preferences = (
        np.array(traj_lens),
        np.array(returns),
        np.array(returns_mo),
        np.array(states),
        np.array(preferences),
    )

    rlogs = {}
    rlogs["n_obj"] = len(trajectories[0]["preference"][0])

    max_prefs = np.max(preferences, axis=0)
    min_prefs = np.min(preferences, axis=0)
    rlogs["dataset_min_prefs"] = min_prefs
    rlogs["dataset_max_prefs"] = max_prefs

    max_raw_r = np.multiply(
        np.max(returns_mo, axis=0), max_prefs
    )  # based on weighted values
    min_raw_r = np.multiply(np.min(returns_mo, axis=0), min_prefs)
    max_final_r = np.max(returns)
    min_final_r = np.min(returns)
    rlogs["dataset_min_raw_r"] = min_raw_r
    rlogs["dataset_max_raw_r"] = max_raw_r
    rlogs["dataset_min_final_r"] = min_final_r
    rlogs["dataset_max_final_r"] = max_final_r

    rollout_weighted_raw_r = None
    rollout_unweighted_raw_r = rollout_original_raw_r = returns_mo
    rlogs["target_returns"] = None
    rlogs["target_prefs"] = None
    rlogs["rollout_unweighted_raw_r"] = rollout_unweighted_raw_r
    rlogs["rollout_weighted_raw_r"] = rollout_weighted_raw_r
    rlogs["rollout_original_raw_r"] = rollout_original_raw_r

    infos = {
        "env": env_name,
        "dataset": datasets[0],  # currently use only one dataset
        "num_traj": num_traj,
    }

    visualize(rlogs, "./experiment_runs/behavior", 0, only_hv_sp=True, infos=infos)


def cal_all():
    for env in [
        "MO-Hopper-v3"
    ]:  # ['MO-Ant-v2', 'MO-HalfCheetah-v2', 'MO-Hopper-v2', 'MO-Hopper-v3', 'MO-Swimmer-v2', 'MO-Walker2d-v2']
        for policy in ["expert", "amateur"]:
            for dist in ["narrow", "uniform", "wide"]:
                dataset = f"{policy}_{dist}"
                num_traj, num_plot = 50000, 1000
                print(f"cal: {env}_{dataset}_{num_traj} use {num_plot} samples...")
                try:
                    cal_from_data([dataset], env)
                except:
                    print("error.")

import argparse
if __name__ == "__main__":
    # cal_all()
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MO-Hopper-v2')
    parser.add_argument('--collect_type', type=str, default="expert")
    # narrow, wide, uniform, custom
    parser.add_argument('--preference_type', type=str, default="custom")
    parser.add_argument('--num_traj', type=int, default=10000)
    parser.add_argument('--num_plot', type=int, default=1000)
    parser.add_argument('--data_path', type=str, default="data_collected")
    parser.add_argument('--p_bar', type=bool, default=False)
    args = parser.parse_args()
    
    dataset = f"{args.collect_type}_{args.preference_type}"
    cal_from_data(datasets=[dataset], env_name=args.env_name, num_traj=args.num_traj, num_plot=args.num_plot, data_path=args.data_path)
