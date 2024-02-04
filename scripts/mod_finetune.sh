# reward_cat vs. rtg_cat
# CUDA_VISIBLE_DEVICES=0 python experiment.py --dir experiment_runs/mod_impl/rtg_cat --env MO-HalfCheetah-v2 --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --seed 1 --dataset expert_uniform --model_type mod --mod_type dd --num_steps_per_iter 20000 --max_iters 2 --use_p_bar True --K 32 --n_diffusion_steps 2 &
# CUDA_VISIBLE_DEVICES=0 python experiment.py --dir experiment_runs/mod_impl/rtg_cat --env MO-HalfCheetah-v2 --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --seed 1 --dataset expert_uniform --model_type mod --mod_type dd --num_steps_per_iter 20000 --max_iters 2 --use_p_bar True --K 32 --n_diffusion_steps 2 --returns_condition True &
# CUDA_VISIBLE_DEVICES=1 python experiment.py --dir experiment_runs/mod_impl/rtg_cat --env MO-HalfCheetah-v2 --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --seed 1 --dataset expert_uniform --model_type mod --mod_type dt --num_steps_per_iter 20000 --max_iters 2 --use_p_bar True --K 32 --n_diffusion_steps 2 &
# CUDA_VISIBLE_DEVICES=1 python experiment.py --dir experiment_runs/mod_impl/rtg_cat --env MO-HalfCheetah-v2 --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --seed 1 --dataset expert_uniform --model_type mod --mod_type dt --num_steps_per_iter 20000 --max_iters 2 --use_p_bar True --K 32 --n_diffusion_steps 2 --returns_condition True &
# wait

# bc diffusion step
CUDA_VISIBLE_DEVICES=0 python experiment.py --dir experiment_runs/mod_impl/finetune/diffussion_step/1 --env MO-HalfCheetah-v2 --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --seed 1 --dataset expert_uniform --model_type mod --mod_type bc --num_steps_per_iter 20000 --max_iters 2 --use_p_bar True --K 32 --n_diffusion_steps 1 &
CUDA_VISIBLE_DEVICES=0 python experiment.py --dir experiment_runs/mod_impl/finetune/diffussion_step/2 --env MO-HalfCheetah-v2 --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --seed 1 --dataset expert_uniform --model_type mod --mod_type bc --num_steps_per_iter 20000 --max_iters 2 --use_p_bar True --K 32 --n_diffusion_steps 2 &
CUDA_VISIBLE_DEVICES=0 python experiment.py --dir experiment_runs/mod_impl/finetune/diffussion_step/4 --env MO-HalfCheetah-v2 --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --seed 1 --dataset expert_uniform --model_type mod --mod_type bc --num_steps_per_iter 20000 --max_iters 2 --use_p_bar True --K 32 --n_diffusion_steps 4 &
CUDA_VISIBLE_DEVICES=1 python experiment.py --dir experiment_runs/mod_impl/finetune/diffussion_step/8 --env MO-HalfCheetah-v2 --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --seed 1 --dataset expert_uniform --model_type mod --mod_type bc --num_steps_per_iter 20000 --max_iters 2 --use_p_bar True --K 32 --n_diffusion_steps 8 &
CUDA_VISIBLE_DEVICES=1 python experiment.py --dir experiment_runs/mod_impl/finetune/diffussion_step/16 --env MO-HalfCheetah-v2 --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --seed 1 --dataset expert_uniform --model_type mod --mod_type bc --num_steps_per_iter 20000 --max_iters 2 --use_p_bar True --K 32 --n_diffusion_steps 16 &
CUDA_VISIBLE_DEVICES=1 python experiment.py --dir experiment_runs/mod_impl/finetune/diffussion_step/32 --env MO-HalfCheetah-v2 --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --seed 1 --dataset expert_uniform --model_type mod --mod_type bc --num_steps_per_iter 20000 --max_iters 2 --use_p_bar True --K 32 --n_diffusion_steps 32 &
wait

# dt diffusion step
CUDA_VISIBLE_DEVICES=0 python experiment.py --dir experiment_runs/mod_impl/finetune/diffussion_step/1 --env MO-HalfCheetah-v2 --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --seed 1 --dataset expert_uniform --model_type mod --mod_type dt --num_steps_per_iter 20000 --max_iters 2 --use_p_bar True --K 32 --n_diffusion_steps 1 &
CUDA_VISIBLE_DEVICES=0 python experiment.py --dir experiment_runs/mod_impl/finetune/diffussion_step/2 --env MO-HalfCheetah-v2 --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --seed 1 --dataset expert_uniform --model_type mod --mod_type dt --num_steps_per_iter 20000 --max_iters 2 --use_p_bar True --K 32 --n_diffusion_steps 2 &
CUDA_VISIBLE_DEVICES=1 python experiment.py --dir experiment_runs/mod_impl/finetune/diffussion_step/4 --env MO-HalfCheetah-v2 --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --seed 1 --dataset expert_uniform --model_type mod --mod_type dt --num_steps_per_iter 20000 --max_iters 2 --use_p_bar True --K 32 --n_diffusion_steps 4 &
CUDA_VISIBLE_DEVICES=1 python experiment.py --dir experiment_runs/mod_impl/finetune/diffussion_step/8 --env MO-HalfCheetah-v2 --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --seed 1 --dataset expert_uniform --model_type mod --mod_type dt --num_steps_per_iter 20000 --max_iters 2 --use_p_bar True --K 32 --n_diffusion_steps 8 &
wait