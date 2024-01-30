DIR="experiment_runs/all_expert_custom"
ITER=2
STEP1=200000

CUDA_VISIBLE_DEVICES=0 python experiment.py --dir $DIR --env MO-Ant-v2 --data_mode _formal --concat_state_pref 0 --concat_rtg_pref 0 --concat_act_pref 0 --mo_rtg True --seed 111 --dataset expert_custom --model_type rvs --num_steps_per_iter $STEP1 --max_iters $ITER --use_p_bar True &
CUDA_VISIBLE_DEVICES=0 python experiment.py --dir $DIR --env MO-Hopper-v2 --data_mode _formal --concat_state_pref 0 --concat_rtg_pref 0 --concat_act_pref 0 --mo_rtg True --seed 111 --dataset expert_custom --model_type rvs --num_steps_per_iter $STEP1 --max_iters $ITER --use_p_bar True &
CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env MO-HalfCheetah-v2 --data_mode _formal --concat_state_pref 0 --concat_rtg_pref 0 --concat_act_pref 0 --mo_rtg True --seed 111 --dataset expert_custom --model_type rvs --num_steps_per_iter $STEP1 --max_iters $ITER --use_p_bar True &
CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env MO-Swimmer-v2 --data_mode _formal --concat_state_pref 0 --concat_rtg_pref 0 --concat_act_pref 0 --mo_rtg True --seed 111 --dataset expert_custom --model_type rvs --num_steps_per_iter $STEP1 --max_iters $ITER --use_p_bar True &
CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env MO-Walker2d-v2 --data_mode _formal --concat_state_pref 0 --concat_rtg_pref 0 --concat_act_pref 0 --mo_rtg True --seed 111 --dataset expert_custom --model_type rvs --num_steps_per_iter $STEP1 --max_iters $ITER --use_p_bar True &
wait

# TODO eval CQL all
