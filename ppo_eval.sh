#!/bin/bash
python -m baselines.run --alg=ppo2 --env=Pendulum-v0 --nminibatches=32 --noptepochs=10 --num_env=12 --num_timesteps=1e5 --save_path=/home/gaoruoyu/PycharmProjects/garage_test/models/ppo_pendulum
echo "agent generate finish"
 
for((num=1;num<30;num++))
do
echo "num=$num"
python -m baselines.run --alg=ppo2 --env=Pendulum-v0 --nminibatches=32 --noptepochs=10 --num_env=12 --num_timesteps=1e5 --save_path=/home/gaoruoyu/PycharmProjects/garage_test/models/ppo_pendulum --load_path=/home/gaoruoyu/PycharmProjects/garage_test/models/ppo_pendulum --log_path=/home/gaoruoyu/PycharmProjects/garage_test/curve_data_cut
python  eval.py --alg=ppo2 --env=Pendulum-v0 --nminibatches=32 --noptepochs=10  --num_timesteps=0 --num_env=1  --load_path=/home/gaoruoyu/PycharmProjects/garage_test/models/ppo_pendulum --play
done


