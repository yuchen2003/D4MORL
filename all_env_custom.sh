DIR="experiment_runs/all_expert_custom"

for MODEL in dt rvs bc cql
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
    
    CUDA_VISIBLE_DEVICES=0 python experiment.py --dir $DIR --env MO-Hopper-v2 --data_mode _formal --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --mo_rtg True --seed 111 --dataset expert_custom --model_type $MODEL --num_steps_per_iter $STEP1 --max_iters $ITER --use_p_bar True &
    # CUDA_VISIBLE_DEVICES=0 python experiment.py --dir $DIR --env MO-Hopper-v3 --data_mode _formal --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --mo_rtg True --seed 111 --dataset expert_custom --model_type $MODEL --num_steps_per_iter $STEP1 --max_iters $ITER --use_p_bar False &
    CUDA_VISIBLE_DEVICES=0 python experiment.py --dir $DIR --env MO-Swimmer-v2 --data_mode _formal --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --mo_rtg True --seed 111 --dataset expert_custom --model_type $MODEL --num_steps_per_iter $STEP2 --max_iters $ITER --use_p_bar False &
    CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env MO-Ant-v2 --data_mode _formal --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --mo_rtg True --seed 111 --dataset expert_custom --model_type $MODEL --num_steps_per_iter $STEP3 --max_iters $ITER --use_p_bar False &
    CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env MO-HalfCheetah-v2 --data_mode _formal --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --mo_rtg True --seed 111 --dataset expert_custom --model_type $MODEL --num_steps_per_iter $STEP4 --max_iters $ITER --use_p_bar False &
    CUDA_VISIBLE_DEVICES=1 python experiment.py --dir $DIR --env MO-Walker2d-v2 --data_mode _formal --concat_state_pref $CSP --concat_rtg_pref $CRP --concat_act_pref $CAP --mo_rtg True --seed 111 --dataset expert_custom --model_type $MODEL --num_steps_per_iter $STEP5 --max_iters $ITER --use_p_bar False &
    wait
done

# dd custom
CUDA_VISIBLE_DEVICES=1 python experiment.py --dir experiment_runs/all_expert_custom --env MO-Swimmer-v2 --data_mode _formal --concat_state_pref 1 --concat_rtg_pref 0 --concat_act_pref 0 --mo_rtg True --seed 1 --dataset expert_custom --model_type dd --num_steps_per_iter 200000 --max_iters 2 --use_p_bar True

# plot
for ENV in Hopper Ant HalfCheetah Walker2d Swimmer
do
    python modt/training/visualizer.py --env_name MO-$ENV-v2 --collect_type expert --preference_type custom --num_traj 10000 --num_plot 1000
done

python modt/training/visualizer.py --env_name MO-Ant-v2 --collect_type expert --preference_type uniform --num_traj 10000 --num_plot 1000 &
python modt/training/visualizer.py --env_name MO-HalfCheetah-v2 --collect_type expert --preference_type uniform --num_traj 10000 --num_plot 1000 &
python modt/training/visualizer.py --env_name MO-Hopper-v2 --collect_type expert --preference_type uniform --num_traj 10000 --num_plot 1000 &
python modt/training/visualizer.py --env_name MO-Swimmer-v2 --collect_type expert --preference_type uniform --num_traj 10000 --num_plot 1000 &
python modt/training/visualizer.py --env_name MO-Walker2d-v2 --collect_type expert --preference_type uniform --num_traj 10000 --num_plot 1000 &
python modt/training/visualizer.py --env_name MO-Hopper-v3 --collect_type expert --preference_type uniform --num_traj 10000 --num_plot 1000 &
wait

python modt/training/visualizer.py --env_name MO-Hopper-v3 --collect_type expert --preference_type uniform --num_traj 10000 --num_plot 1000