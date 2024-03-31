NUM_TRAJ=50000

python collect_env_new.py --env_name MO-Hopper-v2 --collect_type amateur --preference_type custom --ood_type fewshot --num_traj $NUM_TRAJ --p_bar True &
python collect_env_new.py --env_name MO-Hopper-v2 --collect_type expert --preference_type custom --ood_type fewshot --num_traj $NUM_TRAJ &
python collect_env_new.py --env_name MO-HalfCheetah-v2 --collect_type amateur --preference_type custom --ood_type fewshot --num_traj $NUM_TRAJ --p_bar True &
python collect_env_new.py --env_name MO-HalfCheetah-v2 --collect_type expert --preference_type custom --ood_type fewshot --num_traj $NUM_TRAJ &
python collect_env_new.py --env_name MO-Hopper-v3 --collect_type amateur --preference_type custom --ood_type fewshot --num_traj $NUM_TRAJ --p_bar True &
python collect_env_new.py --env_name MO-Hopper-v3 --collect_type expert --preference_type custom --ood_type fewshot --num_traj $NUM_TRAJ &
python collect_env_new.py --env_name MO-Ant-v2 --collect_type amateur --preference_type custom --ood_type fewshot --num_traj $NUM_TRAJ --p_bar True &
python collect_env_new.py --env_name MO-Ant-v2 --collect_type expert --preference_type custom --ood_type fewshot --num_traj $NUM_TRAJ &
python collect_env_new.py --env_name MO-Swimmer-v2 --collect_type amateur --preference_type custom --ood_type fewshot --num_traj $NUM_TRAJ --p_bar True &
python collect_env_new.py --env_name MO-Swimmer-v2 --collect_type expert --preference_type custom --ood_type fewshot --num_traj $NUM_TRAJ &
python collect_env_new.py --env_name MO-Walker2d-v2 --collect_type amateur --preference_type custom --ood_type fewshot --num_traj $NUM_TRAJ --p_bar True &
python collect_env_new.py --env_name MO-Walker2d-v2 --collect_type expert --preference_type custom --ood_type fewshot --num_traj $NUM_TRAJ &
wait