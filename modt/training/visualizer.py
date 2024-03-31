import argparse
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
from data_generation.custom_pref import RejectHole, HOLES, HOLES_v2, HOLES_v3

def visualize(rollout_logs, logsdir, cur_step, draw_behavior=False, infos={}, draw_ood=False):
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
    
    if n_obj == 3:
        hole = HOLES_v3
    elif 'MO-Hopper-v2' in logsdir:
        hole = HOLES_v2
    else:
        hole = HOLES
    rejecthole = RejectHole(*hole)

    indices_wanted = undominated_indices(rollout_unweighted_raw_r, tolerance=0.05)
    n_points = len(indices_wanted)
    edge_colors = [
        "royalblue" if i in indices_wanted else "r"
        for i in range(rollout_unweighted_raw_r.shape[0])
    ]
    face_colors = ["none" for i in range(rollout_unweighted_raw_r.shape[0])]
    
    pref_edge_colors = deepcopy(edge_colors)
    if draw_ood:
        for i, t_pref in enumerate(target_prefs):
            is_ood = (t_pref[0] < dataset_min_prefs[0]) or (t_pref[0] > dataset_max_prefs[0]) or (t_pref in rejecthole)
            if is_ood: # ood
                if infos['is_custom'] == True and (t_pref in rejecthole):
                    if pref_edge_colors[i] == 'r':
                        pref_edge_colors[i] = 'm' # intra-ood, dominated
                    else:
                        pref_edge_colors[i] = 'c' # intra-ood, non-dominated
                else:
                    if pref_edge_colors[i] == 'r': 
                        pref_edge_colors[i] = 'y' # extra-ood, dominated
                    else:
                        pref_edge_colors[i] = 'g' # extra-ood, non-dominated
                

    hv = compute_hypervolume(
        rollout_original_raw_r
    )  # this automatically ignores the dominated points
    indices_wanted_strict = undominated_indices(rollout_original_raw_r, tolerance=0)
    front_return_batch = rollout_original_raw_r[indices_wanted_strict]
    sparsity = compute_sparsity(front_return_batch)

    print(f"hv={hv}, sp={sparsity}, n_points={n_points}")

    fig, axes = plt.subplots(n_obj, 3, constrained_layout=True, figsize=(12, 8))
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
            edgecolors=pref_edge_colors,
        )
        axes[cur_ax].set_xlim([0, max(rollout_original_raw_r[:, 0]) * 1.05])
        axes[cur_ax].set_ylim([0, max(rollout_original_raw_r[:, 1]) * 1.05])
        axes[cur_ax].set_title(f"Obj 0 vs Obj 1")
        axes[cur_ax].set(xlabel="Obj 0", ylabel="Obj 1")
        axes[cur_ax].legend(loc="lower right")
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
            edgecolors=pref_edge_colors,
        )
        axes[cur_ax].set_xlim3d([0, max(rollout_original_raw_r[:, 0]) * 1.05])
        axes[cur_ax].set_ylim3d([0, max(rollout_original_raw_r[:, 1]) * 1.05])
        axes[cur_ax].set_zlim3d([0, max(rollout_original_raw_r[:, 2]) * 1.05])
        axes[cur_ax].set_title(f"Obj 1 vs. Obj 2 vs. Obj 3")
        axes[cur_ax].set(xlabel="Obj 1", ylabel="Obj 2", zlabel="Obj 3")
        axes[cur_ax].legend(loc="lower right")
        cur_ax += 1

    rollout_ratio = rollout_original_raw_r / np.sum(
        rollout_original_raw_r, axis=1, keepdims=True
    )
    axes[cur_ax].scatter(
        target_prefs[:, 0],
        rollout_ratio[:, 0],
        label="pref",
        facecolors=face_colors,
        edgecolors=pref_edge_colors,
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
        label="pref",
        facecolors=face_colors,
        edgecolors=pref_edge_colors,
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
            label="pref",
            facecolors=face_colors,
            edgecolors=pref_edge_colors,
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
        
    return_edge_colors = pref_edge_colors

    using_mo_rtg = False if len(target_returns.shape) == 1 else True
    if using_mo_rtg:
        axes[cur_ax].scatter(
            target_returns[:, 0],
            rollout_weighted_raw_r[:, 0],
            facecolors=face_colors,
            edgecolors=return_edge_colors,
            label="return",
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
            edgecolors=return_edge_colors,
            label="return",
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
                edgecolors=return_edge_colors,
                label="return",
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
            edgecolors=return_edge_colors,
            label="return",
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
        
    rollout_scalar_r = np.sum(rollout_weighted_raw_r, axis=1)
    axes[cur_ax].scatter(
        target_prefs[:, 0],
        rollout_scalar_r,
        facecolors=face_colors,
        edgecolors=return_edge_colors,
        label="scalarised",
    )
    axes[cur_ax].set_xlim([np.min(target_prefs[:, 0]) * 0.95, np.max(target_prefs[:, 0]) * 1.05])
    axes[cur_ax].set_ylim([-5, np.max(rollout_scalar_r) * 1.05])
    axes[cur_ax].set(
        xlabel="target pref0", ylabel="achieved scalar return"
    )
    axes[cur_ax].axvline(
        x=dataset_min_prefs[0],
        ls="--",
    )
    axes[cur_ax].axvline(
        x=dataset_max_prefs[0],
        ls="--",
    )
    axes[cur_ax].legend(loc="lower left")
    axes[cur_ax].set_title(f"Scalarised Return")
    cur_ax += 1
    
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    if not os.path.exists(logsdir):
        os.mkdir(logsdir)
    if not draw_behavior:
        plt.savefig(f"{logsdir}/step={cur_step}_plots.png")
    else:
        env_name = infos["env"]
        dataset = infos["dataset"]
        num_traj = infos["num_traj"]
        plt.savefig(f"{logsdir}/{env_name}_{num_traj}_{dataset}_plots.png")
    plt.close()


def cal_behavior_from_data(
    datasets=["expert_wide"], env_name="MO-Hopper-v2", num_traj=50000, num_plot=1000, data_path="data_collected"
):
    assert num_traj >= num_plot
    generation_path = f"data_generation/{data_path}"
    is_custom = False # NOTE currently this only support one dataset (not multiple custom or not custom mixed)
    for i, d in enumerate(datasets):
        if d.endswith('custom'):
            if env_name == 'MO-Hopper-v3':
                hole = HOLES_v3
            elif env_name == 'MO-Hopper-v2':
                hole = HOLES_v2
            else:
                hole = HOLES
            datasets[i] += f'_{hole}'
            is_custom = True
            print(datasets[i])
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

    # padding a few state trajs with 0 to be as long as others.
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
    rlogs["n_obj"] = pref_dim = len(trajectories[0]["preference"][0])

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

    rollout_unweighted_raw_r = rollout_original_raw_r = returns_mo
    rollout_weighted_raw_r = np.multiply(rollout_unweighted_raw_r, preferences)
    rlogs["target_returns"] = rollout_weighted_raw_r
    rlogs["target_prefs"] = preferences
    rlogs["rollout_unweighted_raw_r"] = rollout_unweighted_raw_r
    rlogs["rollout_weighted_raw_r"] = rollout_weighted_raw_r
    rlogs["rollout_original_raw_r"] = rollout_original_raw_r

    infos = {
        "env": env_name,
        "dataset": datasets[0],  # FIXME currently use only one dataset
        "num_traj": num_traj,
        "is_custom": is_custom,
    }

    visualize(rlogs, "./experiment_runs/behavior1", 0, draw_behavior=True, infos=infos)


def cal_all_behavior():
    for env in ['MO-Ant-v2', 'MO-HalfCheetah-v2', 'MO-Hopper-v2', 'MO-Hopper-v3', 'MO-Swimmer-v2', 'MO-Walker2d-v2']:
        for policy in ["expert", "amateur"]:
            for dist in ["narrow", "uniform", "wide", "custom"]:
                dataset = f"{policy}_{dist}"
                num_traj, num_plot = 50000, 1000
                print(f"cal: {env}_{dataset}_{num_traj} use {num_plot} samples...")
                try:
                    cal_behavior_from_data([dataset], env)
                except:
                    print("error.")

def visu_rollout(dir='experiment_runs/all/dt/normal/MO-Ant-v2/expert_custom/1/logs', model_type='dt', concat_type='normal', env_name='MO-Ant-v2', dataset='expert_custom', seed=1, logs='logs', step=10000, num_traj=50000):
    logsdir = f'{dir}/step={step}_rollout.pkl'
    with open(logsdir, 'rb') as f:
        rollout_logs = pickle.load(f)
    savedir = f'{dir}/ood/'
    is_custom = False
    if dataset.endswith('custom'):
        if env_name == 'MO-Hopper-v3':
            hole = HOLES_v3
        elif env_name == 'MO-Hopper-v2':
            hole = HOLES_v2
        else:
            hole = HOLES
        dataset += f'_{hole}'
        is_custom = True
        print(dataset)
    datapath = f'data_generation/data_collected/{env_name}/{env_name}_{num_traj}_new{dataset}.pkl'
    infos = {
        "env": env_name,
        "dataset": dataset,
        "num_traj": num_traj,
        'is_custom': is_custom,
        'datapath': datapath,
        'eps': 0.02,
        'ret_eps': 50,
    }
    visualize(rollout_logs, savedir, step, infos=infos, draw_ood=True) # allows eps/2 error
    print(f'saved to {savedir}')
    
def visu_all_rollout(dir):
    ''' Recursively find all *.pkl '''
    len_dir = len(dir)
    len1, len2 = len('step='), len('_rollout.pkl')
    for root, dirs, files in os.walk(dir):
        # print(root, dirs, files)
        for f_str in files:
            if f_str.endswith('rollout.pkl'):
                configs = root[len_dir + 1 : ].split('/')[-6:]
                print(configs)
                step_str = f_str[len1 : -len2]
                try:
                    visu_rollout(root, *configs, step=step_str)
                except:
                    print('error.')
    print('***end.***')

if __name__ == "__main__":
    ### visualize all rollout under some dir (path till .../{seed}/logs)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dir', type=str, default='experiment_runs/debug')
    # parser.add_argument('--custom_type', type=str, default='large') # large, small, fewshot
    # args = parser.parse_args()
    # visu_all_rollout(args.dir)
    
    ### plot behavior pf
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MO-Hopper-v3')
    parser.add_argument('--collect_type', type=str, default="expert")
    parser.add_argument('--preference_type', type=str, default="custom") # narrow, wide, uniform, custom
    parser.add_argument('--custom_type', type=str, default='large') # large, small, fewshot
    parser.add_argument('--num_traj', type=int, default=50000)
    parser.add_argument('--num_plot', type=int, default=100)
    parser.add_argument('--data_path', type=str, default="data_collected")
    parser.add_argument('--p_bar', type=bool, default=True)
    args = parser.parse_args()
    dataset = f"{args.collect_type}_{args.preference_type}"
    if args.preference_type == 'custom':
        if args.env_name == 'MO-Hopper-v3':
            hole = HOLES_v3
        elif args.env_name == 'MO-Hopper-v2':
            hole = HOLES_v2
        else:
            hole = HOLES
        dataset += f'_{args.custom_type}_{hole.radius}'
    print("cal behavior:", args.__dict__)
    cal_behavior_from_data(datasets=[dataset], env_name=args.env_name, num_traj=args.num_traj, num_plot=args.num_plot, data_path=args.data_path)
    
    ### plot all behavior pf
    # cal_all_behavior() # FIXME issueing
