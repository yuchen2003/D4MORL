CUDA_VISIBLE_DEVICES=0 python collect_env_new.py --env_name MO-Ant-v2 --collect_type expert --preference_type custom --num_traj 10000 --p_bar True &
CUDA_VISIBLE_DEVICES=0 python collect_env_new.py --env_name MO-HalfCheetah-v2 --collect_type expert --preference_type custom --num_traj 10000 &
CUDA_VISIBLE_DEVICES=0 python collect_env_new.py --env_name MO-Hopper-v2 --collect_type expert --preference_type custom --num_traj 10000 &
CUDA_VISIBLE_DEVICES=1 python collect_env_new.py --env_name MO-Swimmer-v2 --collect_type expert --preference_type custom --num_traj 10000 &
CUDA_VISIBLE_DEVICES=1 python collect_env_new.py --env_name MO-Walker2d-v2 --collect_type expert --preference_type custom --num_traj 10000 &
wait
CUDA_VISIBLE_DEVICES=1 python collect_env_new.py --env_name MO-Hopper-v3 --collect_type expert --preference_type custom --num_traj 10000 & 
wait
