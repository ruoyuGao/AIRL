#!/bin/bash

#python -m baselines.run --alg=ppo2 --env=Pendulum-v0 --nminibatches=32 --noptepochs=10 --num_env=12 --num_timesteps=4e5 --save_path=/home/gaoruoyu/PycharmProjects/garage_test/models/ppo_irl_pendulum
echo "-------------agent generate finish--------------------------------------------------------"
for((num=1; num<10 ;num++))
do
	echo "------------------num=$num----------------------------------------------------------"
	python  learn_reward.py --alg=ppo2 --env=Pendulum-v0 --nminibatches=32 --noptepochs=10 --num_timesteps=0 --num_env=1 --save_path=/home/gaoruoyu/PycharmProjects/garage_test/models/reward  --load_path=/home/gaoruoyu/PycharmProjects/garage_test/models/ppo_irl_pendulum
	echo "----------------learning reward finish---------------------------------------"
	python  ppo_AIRL.py --alg=ppo2 --env=Pendulum-v0 --nminibatches=32 --noptepochs=10 --num_env=12  --num_timesteps=1e5 --save_path=/home/gaoruoyu/PycharmProjects/garage_test/models/ppo_irl_pendulum --load_path=/home/gaoruoyu/PycharmProjects/garage_test/models/ppo_irl_pendulum
	python  eval.py --alg=ppo2 --env=Pendulum-v0 --nminibatches=32 --noptepochs=10  --num_timesteps=0 --num_env=1  --load_path=/home/gaoruoyu/PycharmProjects/garage_test/models/ppo_irl_pendulum --play
	echo "---------------evaluated------------------------------------------------------"

done
	
