python collect_env_new.py --env_name MO-Hopper-v2 --collect_type amateur --preference_type uniform --num_traj 50000 --p_bar True &
python collect_env_new.py --env_name MO-Hopper-v2 --collect_type expert --preference_type uniform --num_traj 50000 &
python collect_env_new.py --env_name MO-Hopper-v2 --collect_type amateur --preference_type wide --num_traj 50000 &
python collect_env_new.py --env_name MO-Hopper-v2 --collect_type expert --preference_type wide --num_traj 50000 &
python collect_env_new.py --env_name MO-Hopper-v2 --collect_type amateur --preference_type narrow --num_traj 50000 &
python collect_env_new.py --env_name MO-Hopper-v2 --collect_type expert --preference_type narrow --num_traj 50000 &
python collect_env_new.py --env_name MO-Hopper-v2 --collect_type amateur --preference_type custom --num_traj 50000 &
python collect_env_new.py --env_name MO-Hopper-v2 --collect_type expert --preference_type custom --num_traj 50000 &
wait

python collect_env_new.py --env_name MO-Ant-v2 --collect_type amateur --preference_type uniform --num_traj 50000 --p_bar True &
python collect_env_new.py --env_name MO-Ant-v2 --collect_type expert --preference_type uniform --num_traj 50000 &
python collect_env_new.py --env_name MO-Ant-v2 --collect_type amateur --preference_type wide --num_traj 50000 &
python collect_env_new.py --env_name MO-Ant-v2 --collect_type expert --preference_type wide --num_traj 50000 &
python collect_env_new.py --env_name MO-Ant-v2 --collect_type amateur --preference_type narrow --num_traj 50000 &
python collect_env_new.py --env_name MO-Ant-v2 --collect_type expert --preference_type narrow --num_traj 50000 &
wait

python collect_env_new.py --env_name MO-HalfCheetah-v2 --collect_type amateur --preference_type uniform --num_traj 50000 --p_bar True &
python collect_env_new.py --env_name MO-HalfCheetah-v2 --collect_type expert --preference_type uniform --num_traj 50000 &
python collect_env_new.py --env_name MO-HalfCheetah-v2 --collect_type amateur --preference_type wide --num_traj 50000 &
python collect_env_new.py --env_name MO-HalfCheetah-v2 --collect_type expert --preference_type wide --num_traj 50000 &
python collect_env_new.py --env_name MO-HalfCheetah-v2 --collect_type amateur --preference_type narrow --num_traj 50000 &
python collect_env_new.py --env_name MO-HalfCheetah-v2 --collect_type expert --preference_type narrow --num_traj 50000 &
wait

python collect_env_new.py --env_name MO-Swimmer-v2 --collect_type amateur --preference_type uniform --num_traj 50000 --p_bar True &
python collect_env_new.py --env_name MO-Swimmer-v2 --collect_type expert --preference_type uniform --num_traj 50000 &
python collect_env_new.py --env_name MO-Swimmer-v2 --collect_type amateur --preference_type wide --num_traj 50000 &
python collect_env_new.py --env_name MO-Swimmer-v2 --collect_type expert --preference_type wide --num_traj 50000 &
python collect_env_new.py --env_name MO-Swimmer-v2 --collect_type amateur --preference_type narrow --num_traj 50000 &
python collect_env_new.py --env_name MO-Swimmer-v2 --collect_type expert --preference_type narrow --num_traj 50000 &
wait

python collect_env_new.py --env_name MO-Walker2d-v2 --collect_type amateur --preference_type uniform --num_traj 50000 --p_bar True &
python collect_env_new.py --env_name MO-Walker2d-v2 --collect_type expert --preference_type uniform --num_traj 50000 &
python collect_env_new.py --env_name MO-Walker2d-v2 --collect_type amateur --preference_type wide --num_traj 50000 &
python collect_env_new.py --env_name MO-Walker2d-v2 --collect_type expert --preference_type wide --num_traj 50000 &
python collect_env_new.py --env_name MO-Walker2d-v2 --collect_type amateur --preference_type narrow --num_traj 50000 &
python collect_env_new.py --env_name MO-Walker2d-v2 --collect_type expert --preference_type narrow --num_traj 50000 &
wait

python collect_env_new.py --env_name MO-Hopper-v3 --collect_type amateur --preference_type uniform --num_traj 50000 --p_bar True &
python collect_env_new.py --env_name MO-Hopper-v3 --collect_type expert --preference_type uniform --num_traj 50000 &
python collect_env_new.py --env_name MO-Hopper-v3 --collect_type amateur --preference_type wide --num_traj 50000 &
python collect_env_new.py --env_name MO-Hopper-v3 --collect_type expert --preference_type wide --num_traj 50000 &
python collect_env_new.py --env_name MO-Hopper-v3 --collect_type amateur --preference_type narrow --num_traj 50000 &
python collect_env_new.py --env_name MO-Hopper-v3 --collect_type expert --preference_type narrow --num_traj 50000 &
wait

# for smoke test
# python collect_env_new.py --env_name MO-Hopper-v3 --collect_type amateur --preference_type uniform --num_traj 50 --p_bar True

# python collect_env_new.py --env_name MO-Ant-v2 --collect_type expert --preference_type uniform --num_traj 1000 --p_bar True
# python collect_env_new.py --env_name MO-Ant-v2 --collect_type expert --preference_type custom --num_traj 1000 --p_bar True

# visualize data distribution
# python modt/training/visualizer.py --env_name MO-Ant-v2 --collect_type expert --preference_type uniform --num_traj 1000 --num_plot 1000

python collect_env_new.py --env_name MO-Hopper-v3 --collect_type amateur --preference_type uniform --num_traj 10000 --p_bar True &
python collect_env_new.py --env_name MO-Hopper-v3 --collect_type expert --preference_type uniform --num_traj 10000 &
python collect_env_new.py --env_name MO-Hopper-v3 --collect_type amateur --preference_type wide --num_traj 10000 &
python collect_env_new.py --env_name MO-Hopper-v3 --collect_type expert --preference_type wide --num_traj 10000 &
python collect_env_new.py --env_name MO-Hopper-v3 --collect_type amateur --preference_type narrow --num_traj 10000 &
python collect_env_new.py --env_name MO-Hopper-v3 --collect_type expert --preference_type narrow --num_traj 10000 &
wait

python collect_env_new.py --env_name MO-Hopper-v2 --collect_type amateur --preference_type custom --num_traj 10000 --p_bar True &
python collect_env_new.py --env_name MO-Hopper-v2 --collect_type expert --preference_type custom --num_traj 10000 &
wait