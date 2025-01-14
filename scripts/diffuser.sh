DIR="experiment_runs/mod_final"

for ENV in MO-HalfCheetah-v2 MO-Hopper-v2 MO-Hopper-v3 MO-Swimmer-v2 MO-Walker2d-v2
do
    if [ "$ENV" = "MO-HalfCheetah-v2" ]; then
        STEP=400000
        MNUM=6
        MSTEP=400000
    elif [ "$ENV" = "MO-Hopper-v2" ]; then
        STEP=400000
        MNUM=6
        MSTEP=50000
    elif [ "$ENV" = "MO-Hopper-v3" ]; then
        STEP=200000
        MNUM=5
        MSTEP=100000
    elif [ "$ENV" = "MO-Swimmer-v2" ]; then
        STEP=200000
        MNUM=5
        MSTEP=100000
    elif [ "$ENV" = "MO-Walker2d-v2" ]; then
        STEP=400000
        MNUM=6
        MSTEP=150000
    fi

    for QUALITY in expert amateur
    do
        for SEED in 1 2 3
        do  
        {
            CUDA_VISIBLE_DEVICES=0 python experiment.py --dir $DIR --env $ENV --seed $SEED --dataset $QUALITY'_'uniform --model_type mod --mod_type bc --num_steps_per_iter $STEP --max_iters 1 --use_p_bar True --K 8 --infer_N 7 --n_diffusion_steps 8 --returns_condition True --mixup True --mixup_num $MNUM --mixup_step $MSTEP &
            CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env $ENV --seed $SEED --dataset $QUALITY'_'wide --model_type mod --mod_type bc --num_steps_per_iter $STEP --max_iters 1 --use_p_bar True --K 8 --infer_N 7 --n_diffusion_steps 8 --returns_condition True --mixup True --mixup_num $MNUM --mixup_step $MSTEP &
            CUDA_VISIBLE_DEVICES=0 python experiment.py --dir $DIR --env $ENV --seed $SEED --dataset $QUALITY'_'narrow --model_type mod --mod_type bc --num_steps_per_iter $STEP --max_iters 1 --use_p_bar True --K 8 --infer_N 7 --n_diffusion_steps 8 --returns_condition True --mixup True --mixup_num $MNUM --mixup_step $MSTEP &
            CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env $ENV --seed $SEED --dataset $QUALITY'_'custom --model_type mod --mod_type bc --num_steps_per_iter $STEP --max_iters 1 --use_p_bar True --K 8 --infer_N 7 --n_diffusion_steps 8 --returns_condition True --mixup True --mixup_num $MNUM --mixup_step $MSTEP
        } &
        done
        wait
    done
done

STEP=100000
MNUM=8
MSTEP=100000
for SEED in 1 2 3
do  
    CUDA_VISIBLE_DEVICES=0 python experiment.py --dir $DIR --env MO-Ant-v2 --seed $SEED --dataset amateur_uniform --model_type mod --mod_type bc --num_steps_per_iter $STEP --max_iters 1 --use_p_bar True --K 8 --infer_N 7 --n_diffusion_steps 8 --returns_condition True --mixup True --mixup_num $MNUM --mixup_step $MSTEP &
    CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env MO-Ant-v2 --seed $SEED --dataset expert_uniform --model_type mod --mod_type bc --num_steps_per_iter $STEP --max_iters 1 --use_p_bar True --K 8 --infer_N 7 --n_diffusion_steps 8 --returns_condition True --mixup True --mixup_num $MNUM --mixup_step $MSTEP &
    CUDA_VISIBLE_DEVICES=0 python experiment.py --dir $DIR --env MO-Ant-v2 --seed $SEED --dataset amateur_wide --model_type mod --mod_type bc --num_steps_per_iter $STEP --max_iters 1 --use_p_bar True --K 8 --infer_N 7 --n_diffusion_steps 8 --returns_condition True --mixup True --mixup_num $MNUM --mixup_step $MSTEP &
    CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env MO-Ant-v2 --seed $SEED --dataset expert_wide --model_type mod --mod_type bc --num_steps_per_iter $STEP --max_iters 1 --use_p_bar True --K 8 --infer_N 7 --n_diffusion_steps 8 --returns_condition True --mixup True --mixup_num $MNUM --mixup_step $MSTEP &
    CUDA_VISIBLE_DEVICES=0 python experiment.py --dir $DIR --env MO-Ant-v2 --seed $SEED --dataset amateur_narrow --model_type mod --mod_type bc --num_steps_per_iter $STEP --max_iters 1 --use_p_bar True --K 8 --infer_N 7 --n_diffusion_steps 8 --returns_condition True --mixup True --mixup_num $MNUM --mixup_step $MSTEP &
    CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env MO-Ant-v2 --seed $SEED --dataset expert_narrow --model_type mod --mod_type bc --num_steps_per_iter $STEP --max_iters 1 --use_p_bar True --K 8 --infer_N 7 --n_diffusion_steps 8 --returns_condition True --mixup True --mixup_num $MNUM --mixup_step $MSTEP &
    CUDA_VISIBLE_DEVICES=0 python experiment.py --dir $DIR --env MO-Ant-v2 --seed $SEED --dataset amateur_custom --model_type mod --mod_type bc --num_steps_per_iter $STEP --max_iters 1 --use_p_bar True --K 8 --infer_N 7 --n_diffusion_steps 8 --returns_condition True --mixup True --mixup_num $MNUM --mixup_step $MSTEP &
    CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env MO-Ant-v2 --seed $SEED --dataset expert_custom --model_type mod --mod_type bc --num_steps_per_iter $STEP --max_iters 1 --use_p_bar True --K 8 --infer_N 7 --n_diffusion_steps 8 --returns_condition True --mixup True --mixup_num $MNUM --mixup_step $MSTEP &
    wait
done
