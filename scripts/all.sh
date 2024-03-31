DIR="experiment_runs/all"

for TYPE in naive # normal naive
do
    for MODEL in dt bc # dt rvs bc cql
    do
        for ENV in MO-Hopper-v2 MO-Swimmer-v2 MO-Ant-v2 MO-HalfCheetah-v2 MO-Walker2d-v2 MO-Hopper-v3
        do
            if [ "$MODEL" = "rvs" ]; then
                STEP=100000
                ITER=2
                CSP=1
                CAP=0
                CRP=0
            elif [ "$MODEL" = "dt" ]; then
                ITER=2
                CSP=1
                CAP=1
                CRP=1
                if [ "$ENV" = "MO-Hopper-v2" ] || [ "$ENV" = "MO-Hopper-v3" ]; then
                    STEP=200000
                elif [ "$ENV" = "MO-Swimmer-v2" ]; then
                    STEP=130000
                elif [ "$ENV" = "MO-Ant-v2" ]; then
                    STEP=10000
                elif [ "$ENV" = "MO-HalfCheetah-v2" ]; then
                    STEP=40000
                elif [ "$ENV" = "MO-Walker2d-v2" ]; then
                    STEP=180000
                fi
            elif [ "$MODEL" = "bc" ]; then
                STEP=100000
                ITER=2
                CSP=1
                CAP=0
                CRP=0
            elif [ "$MODEL" = "cql"]; then
                STEP=200000
                ITER=2
                CSP=1
                CAP=0
                CRP=0
            fi
            # if naive, change respective concat
            if [ "$TYPE" = "naive" ]; then
                CSP=0
                CAP=0
                CRP=0
            fi
            echo "***" $TYPE $MODEL $ENV $CSP $CRP $CAP $STEP "***"
            CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env $ENV --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --seed 1 --dataset expert_custom --model_type $MODEL --num_steps_per_iter $STEP --max_iters $ITER &
            CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env $ENV --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --seed 1 --dataset expert_uniform --model_type $MODEL --num_steps_per_iter $STEP --max_iters $ITER &
            CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env $ENV --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --seed 1 --dataset expert_wide --model_type $MODEL --num_steps_per_iter $STEP --max_iters $ITER --use_p_bar False &
            CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env $ENV --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --seed 1 --dataset expert_narrow --model_type $MODEL --num_steps_per_iter $STEP --max_iters $ITER --use_p_bar False &
            CUDA_VISIBLE_DEVICES=0 python experiment.py --dir $DIR --env $ENV --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --seed 1 --dataset amateur_custom --model_type $MODEL --num_steps_per_iter $STEP --max_iters $ITER --use_p_bar False &
            CUDA_VISIBLE_DEVICES=0 python experiment.py --dir $DIR --env $ENV --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --seed 1 --dataset amateur_uniform --model_type $MODEL --num_steps_per_iter $STEP --max_iters $ITER --use_p_bar False &
            CUDA_VISIBLE_DEVICES=0 python experiment.py --dir $DIR --env $ENV --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --seed 1 --dataset amateur_wide --model_type $MODEL --num_steps_per_iter $STEP --max_iters $ITER --use_p_bar False &
            CUDA_VISIBLE_DEVICES=0 python experiment.py --dir $DIR --env $ENV --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --seed 1 --dataset amateur_narrow --model_type $MODEL --num_steps_per_iter $STEP --max_iters $ITER --use_p_bar False &
            wait
        done
    done
done

exit

# CUDA_VISIBLE_DEVICES=1 python experiment.py --dir experiment_runs/all2 --env MO-Hopper-v3 --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --seed 1 --dataset expert_custom --model_type bc --num_steps_per_iter 100000 --max_iters 2 &
# CUDA_VISIBLE_DEVICES=1 python experiment.py --dir experiment_runs/all2 --env MO-Hopper-v3 --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --seed 1 --dataset expert_custom --model_type rvs --num_steps_per_iter 100000 --max_iters 2 &
# CUDA_VISIBLE_DEVICES=0 python experiment.py --dir experiment_runs/all2 --env MO-Hopper-v3 --concat_state_pref 1 --concat_rtg_pref 1 --concat_act_pref 1 --seed 1 --dataset expert_custom --model_type dt --num_steps_per_iter 200000 --max_iters 2 &
# wait

# CUDA_VISIBLE_DEVICES=1 python experiment.py --dir experiment_runs/all --env MO-Humanoid-v2 --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --seed 1 --dataset expert_uniform --num_traj 5000 --model_type rvs --num_steps_per_iter 100000 --max_iters 2 --embed_dim 1024