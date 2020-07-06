import gym
import numpy as np
import pickle
import tensorflow as tf
env=gym.make('BreakoutNoFrameskip-v4')
obs=env.reset()
print(np.size(obs))
print(env.observation_space)
print(env.action_space)

times=500
file_path='/home/gaoruoyu/PycharmProjects/garage_test/data/expert_data_observations_ppo_500.data'
with open(file_path,'rb') as f1:
    list_obs=pickle.load(f1)

print(np.mean(list_obs))
#with open('expert_data_action_ppo.data','rb') as f2:
#    list_action=pickle.load(f2)
#print(np.shape(list_obs))
#print(np.shape(list_action))
#labels_expert=np.ones(10000)

#observation=tf.convert_to_tensor(np.ones([84,84,4]),dtype=tf.int32)
#observation=tf.expand_dims(observation,0)
#actions=tf.convert_to_tensor([5],dtype=tf.int32)
#action_2=tf.expand_dims(actions,0)
#action_3=tf.expand_dims(action_2,0)
#action_3=tf.expand_dims(action_3,0)
#print(action_3.shape)
#new_input=tf.concat([observation,action_3],1)
#print(new_input.shape)

#obss=np.reshape(list_obs,[-1,84,84,4]).astype(float)
#print('finish')
#model=tf.keras.models.load_model('/home/gaoruoyu/PycharmProjects/garage_test/models/reward')
#output=model(obss[:10])

#print(output.numpy())

x=np.array([1,0,0,0,1,0,1,0,0,1])
x=[ 0.9*n if n==1 else n+0.1 for n in x]
print(x)