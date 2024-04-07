import numpy as np
import wandb
import argparse
import os
import json
import time
from time import sleep

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MO-Hopper-v2')
    parser.add_argument('--dataset', type=str, nargs='+', default=['expert_uniform'])
    parser.add_argument('--num_traj', type=int, default=50000)
    parser.add_argument('--data_mode', type=str, default='_formal')
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20) # trajectory horizon
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt, bc, rvs, cql, mod
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=3) # lamb's default should be 4
    parser.add_argument('--n_head', type=int, default=1) # lamb's default should be 4
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=2e-4) # just for mod
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-3)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=1)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--num_steps_per_iter', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dir', type=str, default='experiment_runs')
    parser.add_argument('--log_to_wandb', type=bool, default=False)
    parser.add_argument('--wandb_group', type=str, default='none')
    parser.add_argument('--use_obj', type=int, default=-1) # decay to only 1-obj scenario. -1 default means nothing is decayed
    parser.add_argument('--percent_dt', type=float, default=1) # make DT to only use top% of data, default would be 99%
    parser.add_argument('--use_pref_predict_action', type=bool, default=False)
    parser.add_argument('--concat_state_pref', type=int, default=0) #   |
    parser.add_argument('--concat_rtg_pref', type=int, default=0)   #   | }-> w/, w/o pref (P)
    parser.add_argument('--concat_act_pref', type=int, default=0)   #   |
    parser.add_argument('--normalize_reward', type=bool, default=False)
    parser.add_argument('--mo_rtg', type=bool, default=True)
    parser.add_argument('--eval_only', type=bool, default=False) # may not suitable for continuing training
    parser.add_argument('--return_loss', type=bool, default=False)
    parser.add_argument('--pref_loss', type=bool, default=False)
    parser.add_argument('--optimizer', type=str, default="adam") # adam, lamb
    parser.add_argument('--eval_context_length', type=int, default=5)
    parser.add_argument('--rtg_scale', type=float, default=1)
    parser.add_argument('--seed', type=int, default=123454321)
    parser.add_argument('--granularity', type=int, default=500) # or 18 for hopper3d (324 points)
    parser.add_argument('--use_max_rtg', type=bool, default=False)
    parser.add_argument('--use_p_bar', type=bool, default=True)
    parser.add_argument('--conservative_q', type=int, default=3)
    # MODiffuser configs
    parser.add_argument('--mod_type', type=str, default='bc') # bc, dd, dt
    parser.add_argument('--infer_N', type=int, default=-1) # traj_gen = tau_{t-M+1:t} (M cond) ## tau_{t+1:t+N} (N infer); notice a_hat = a_t
    parser.add_argument('--n_diffusion_steps', type=int, default=10)
    parser.add_argument('--returns_condition', type=bool, default=False) # if want to set False, just not use this option
    parser.add_argument('--v_cfg_w', type=float, default=0.1)
    parser.add_argument('--concat_on', type=str, default='r') # g, r
    parser.add_argument('--diffuser_sample_verbose', type=bool, default=False)
    parser.add_argument('--collect', type=bool, default=False)
    parser.add_argument('--collect_num', type=int, default=10000)
    parser.add_argument('--mixup', type=bool, default=False)
    parser.add_argument('--mixup_step', type=int, default=100000)
    parser.add_argument('--mixup_num', type=int, default=6)
    
    args = parser.parse_args()
    
    seed = args.seed if args.seed is not None else np.random.randint(0, 100000)
    # seed_everything(seed=seed)
    
    dataset_name = '+'.join(args.dataset)
    # if 'custom' in dataset_name: dataset_name += '_' + TAG
    
    if args.concat_state_pref + args.concat_act_pref + args.concat_rtg_pref == 0:
        typ = 'naive'
    else:
        typ = 'normal'
    if args.model_type == 'mod':
        typ += f'/{args.mod_type}'
        
    if args.mixup == False: args.mixup_num = 0
        
    args.run_name = f"{args.dir}/{args.model_type}/{typ}/{args.env}/{dataset_name}/{args.seed}"
    args.dir = args.run_name
    args.run_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        
    if not os.path.exists(args.run_name):
        os.makedirs(args.run_name)
    with open(args.run_name + '/config.json', 'w') as f:
        json_str = json.dumps(vars(args), indent=2)
        
        if args.model_type == 'cql':
            from modt.models.cql import CQL_config
            json_str_2 = ',\n\n' + json.dumps(CQL_config, indent=2)
        else:
            json_str_2 = ',\n\n'
            
        f.write('[' + json_str + json_str_2)

    if args.log_to_wandb:
        wandb.init(
            project=args.wandb_group,
            entity="baitingz",
            name=args.run_name
        )

    print('start', args.env, args.dataset, args.model_type, args.mod_type, args.dir, args.mixup_step)
    if args.env == 'MO-Ant-v2':
        sleep(1)
    else:
        sleep(1)
    print('end', args.env, args.dataset, args.model_type, args.mod_type, args.dir, args.mixup_step)
    # experiment(variant=vars(args))
