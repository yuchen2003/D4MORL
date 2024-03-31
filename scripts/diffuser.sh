DIR="experiment_runs/mod_final"

# 6env parallel (wait) x 2 quality parallel (wait) x 3 or 4 dist x 3 seed
for ENV in MO-Ant-v2 MO-HalfCheetah-v2 # MO-Hopper-v2 MO-Hopper-v3 MO-Swimmer-v2 MO-Walker2d-v2
do
{
    if [ "$ENV" = "MO-Ant-v2" ]; then
        STEP=100000
        MNUM=8
        MSTEP=100000
    elif [ "$ENV" = "MO-HalfCheetah-v2" ]; then
        STEP=200000
        MNUM=7
        MSTEP=100000
    elif [ "$ENV" = "MO-Hopper-v2" ]; then
        STEP=400000
        MNUM=6
        MSTEP=50000
    elif [ "$ENV" = "MO-Hopper-v3" ]; then
        STEP=100000 # ?
        MNUM=5
        MSTEP=100000
    elif [ "$ENV" = "MO-Swimmer-v2" ]; then
        STEP=100000 # ?
        MNUM=8
        MSTEP=100000
    elif [ "$ENV" = "MO-Walker2d-v2" ]; then
        STEP=400000
        MNUM=6
        MSTEP=150000
    fi

    for DIST in uniform custom narrow wide
    do
        for SEED in 1 2 3
        do  
            CUDA_VISIBLE_DEVICES=0 python experiment1.py --dir $DIR --env $ENV --seed $SEED --dataset amateur'_'$DIST --model_type mod --mod_type bc --num_steps_per_iter $STEP --max_iters 1 --use_p_bar True --K 8 --infer_N 7 --n_diffusion_steps 8 --returns_condition True --mixup True --mixup_num $MNUM --mixup_step $MSTEP &
            CUDA_VISIBLE_DEVICES=1 python experiment1.py --dir $DIR --env $ENV --seed $SEED --dataset expert'_'$DIST --model_type mod --mod_type bc --num_steps_per_iter $STEP --max_iters 1 --use_p_bar True --K 8 --infer_N 7 --n_diffusion_steps 8 --returns_condition True --mixup True --mixup_num $MNUM --mixup_step $MSTEP &
            wait
        done
    done

} &
done
wait
