DIR='aaa'

for ENV in MO-Ant-v2 MO-HalfCheetah-v2 MO-Hopper-v2
do
{
    STEP=400000

    for QUALITY in expert amateur
    do
        for DIST in uniform custom narrow wide
        do
            for SEED in 1 2 3
            do  
                CUDA_VISIBLE_DEVICES=0 python experiment1.py --dir $DIR --env $ENV --seed $SEED --dataset $QUALITY'_'$DIST --model_type rvs --concat_state_pref 1 --num_steps_per_iter $STEP --max_iters 1 --use_p_bar True
            done
        done
    done
} &
done

for ENV in MO-Hopper-v3 MO-Swimmer-v2 MO-Walker2d-v2
do
{
    STEP=400000

    for QUALITY in expert amateur
    do
        for DIST in uniform custom narrow wide
        do
            for SEED in 1 2 3
            do  
                CUDA_VISIBLE_DEVICES=1 python experiment1.py --dir $DIR --env $ENV --seed $SEED --dataset $QUALITY'_'$DIST --model_type rvs --concat_state_pref 1 --num_steps_per_iter $STEP --max_iters 1 --use_p_bar True
            done
        done
    done
} &
done
wait