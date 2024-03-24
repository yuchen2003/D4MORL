DIR="experiment_runs/all_expert_custom"

for MODEL in mod rvs dt bc cql
do
    if [ "$MODEL" = "dt" ]; then
        CSP=1
        CAP=1
        CRP=1
        STEP1=200000
        STEP2=130000
        STEP3=10000
        STEP4=40000
        STEP5=18000
    else if [ "$MODEL" = "mod" ]; then
        CSP=0
        CAP=0
        CRP=0
        STEP1=200000
        STEP2=200000
        STEP3=200000
        STEP4=200000
        STEP5=200000
    else
        CSP=1
        CAP=0
        CRP=0
        STEP1=200000
        STEP2=200000
        STEP3=200000
        STEP4=200000
        STEP5=200000
    fi
    
    ITER=2
    
    CUDA_VISIBLE_DEVICES=0 python experiment.py --dir $DIR --env MO-Hopper-v2 --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --seed 111 --dataset expert_custom --model_type $MODEL --num_steps_per_iter $STEP1 --max_iters $ITER --use_p_bar True &
    CUDA_VISIBLE_DEVICES=0 python experiment.py --dir $DIR --env MO-Hopper-v3 --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --seed 111 --dataset expert_custom --model_type $MODEL --num_steps_per_iter $STEP1 --max_iters $ITER --use_p_bar False &
    CUDA_VISIBLE_DEVICES=0 python experiment.py --dir $DIR --env MO-Swimmer-v2 --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --seed 111 --dataset expert_custom --model_type $MODEL --num_steps_per_iter $STEP2 --max_iters $ITER --use_p_bar False &
    CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env MO-Ant-v2 --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --seed 111 --dataset expert_custom --model_type $MODEL --num_steps_per_iter $STEP3 --max_iters $ITER --use_p_bar False &
    CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env MO-HalfCheetah-v2 --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --seed 111 --dataset expert_custom --model_type $MODEL --num_steps_per_iter $STEP4 --max_iters $ITER --use_p_bar False &
    CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env MO-Walker2d-v2 --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --seed 111 --dataset expert_custom --model_type $MODEL --num_steps_per_iter $STEP5 --max_iters $ITER --use_p_bar False &
    wait
done

# CUDA_VISIBLE_DEVICES=0 python experiment.py --dir experiment_runs/all_expert_custom --env MO-Hopper-v3 --data_mode _formal --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --mo_rtg True --seed 111 --dataset expert_custom --model_type dt --num_steps_per_iter 200000 --max_iters 2 --use_p_bar True &
# CUDA_VISIBLE_DEVICES=0 python experiment.py --dir experiment_runs/all_expert_custom --env MO-Hopper-v3 --data_mode _formal --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --mo_rtg True --seed 111 --dataset expert_custom --model_type bc --num_steps_per_iter 200000 --max_iters 2 --use_p_bar True &
# CUDA_VISIBLE_DEVICES=1 python experiment.py --dir experiment_runs/all_expert_custom --env MO-Hopper-v3 --data_mode _formal --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --mo_rtg True --seed 111 --dataset expert_custom --model_type cql --num_steps_per_iter 200000 --max_iters 2 --use_p_bar True &
# CUDA_VISIBLE_DEVICES=1 python experiment.py --dir experiment_runs/all_expert_custom --env MO-Hopper-v3 --data_mode _formal --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --mo_rtg True --seed 111 --dataset expert_custom --model_type rvs --num_steps_per_iter 200000 --max_iters 2 --use_p_bar True &
# wait

# # mod custom
# CUDA_VISIBLE_DEVICES=1 python experiment.py --dir experiment_runs/all_expert_custom --env MO-Swimmer-v2 --concat_state_pref --concat_rtg_pref 0 --concat_act_pref 0 --seed 1 --dataset expert_custom --model_type mod --num_steps_per_iter 200000 --max_iters 2 --use_p_bar True --mixup True --mixup_num 16
