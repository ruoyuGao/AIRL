import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

from baselines.run import *
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras import Model

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}

class simple_model(Model):
    def __init__(self):
        super(simple_model, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten=Flatten()
        self.dense1=Dense(512,kernel_initializer='random_uniform')
        self.dense2=Dense(1,activation=tf.math.sigmoid,kernel_initializer='random_uniform')
    def call(self,x):
        #x=self.conv1(x)
        x=self.flatten(x)
        x=self.dense1(x)
        return self.dense2(x)

class Reward_model(Model):
    def __init__(self):
        super(Reward_model,self).__init__()
        self.conv1=Conv2D(64,3,activation='relu',kernel_initializer='random_uniform')
        self.mp1=MaxPooling2D(2)
        self.conv2 = Conv2D(32, 3,kernel_initializer='random_uniform')
        self.mp2 = MaxPooling2D(2)
        self.conv3 = Conv2D(16, 3,kernel_initializer='random_uniform')
        self.flatten=Flatten()
        self.dense1=Dense(512,kernel_initializer='random_uniform')
        self.dense2=Dense(1,activation=tf.math.sigmoid,kernel_initializer='random_uniform')


    def call(self,x):
        x=self.conv1(x)
        x=self.mp1(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x=self.conv3(x)
        x=self.flatten(x)
        x=self.dense1(x)
        return self.dense2(x)


@tf.function
def train_step(input,labels,train_loss,train_accuracy):
    with tf.GradientTape() as tape:
        predictions=rew_model(input)
        loss=loss_object(labels,predictions)
    gradient=tape.gradient(loss,rew_model.trainable_variables)
    optimizer.apply_gradients(zip(gradient,rew_model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels,test_loss,test_accuracy):
  predictions = rew_model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

def create_dataset_breakout(obs_a,obs_e,n_s=10000,batch_size=32):
    expert_label=np.ones(np.shape(obs_e)[0]*np.shape(obs_e)[1])
    agent_label=np.zeros(np.shape(obs_a)[0]*np.shape(obs_a)[1])
    observation_expert=np.reshape(obs_e,[-1,84,84,4])
    observation_agent=np.reshape(obs_a,[-1,84,84,4])
    obs=tf.concat([observation_expert,observation_agent],0)
    obs=tf.dtypes.cast(obs,float)
    label=tf.concat([expert_label,agent_label],0)
    dataset=tf.data.Dataset.from_tensor_slices((obs,label)).shuffle(n_s).batch(batch_size)
    print('dataset have been created')
    return dataset


def generate_agent_data(agent_model,env,number):
    observation_data_agent=[]
    action_data_agent=[]
    episode_reward=[]
    obs=env.reset()
    episode_rew=0
    state = agent_model.initial_state if hasattr(agent_model, 'initial_state') else None

    for n in range(number):
        if state is not None:
            actions, _, state, _ = agent_model.step(obs)
        else:
            actions, _, _, _ = agent_model.step(obs)

        obs, rew, done, _ = env.step(actions.numpy())
        observation_data_agent.append(obs)
        action_data_agent.append(actions)
        episode_rew+=rew
        env.render()
        if done:
            episode_reward.append(episode_rew)
            episode_rew=0
    print(np.mean(episode_reward))
    print(np.shape(observation_data_agent))
    file_path = '/home/gaoruoyu/PycharmProjects/garage_test/data/agent_data_observation_ppo_{}.data'
    with open(file_path.format(number), 'wb') as f1:
        pickle.dump(observation_data_agent,f1)
    return observation_data_agent



def main(args):
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    loss_list=[]
    epoch_list=[]

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(args.log_path, format_strs=[])

    agent_model,env=train(args,extra_args)




    #get dataset
    obs_a = generate_agent_data(agent_model, env,24000)
    obs_at=generate_agent_data(agent_model,env,6000)
    env.close()

    with open('/home/gaoruoyu/PycharmProjects/garage_test/data/expert_data_observations_ppo_2000.data', 'rb') as f1:
        obs_e = pickle.load(f1)
    dat = create_dataset_breakout(obs_a, obs_e)
    with open('/home/gaoruoyu/PycharmProjects/garage_test/data/expert_data_observations_ppo_500.data', 'rb') as f1:
        obs_et = pickle.load(f1)
    test_dat=create_dataset_breakout(obs_at,obs_et)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    for epoch in range(num_epochs):
        for observation,label in dat:
            train_step(observation,label,train_loss,train_accuracy)

        template = 'Epoch {}, Loss: {},Accuracy: {} ,Test Loss: {}, Test Accuracy: {}'
        for test_images, test_labels in test_dat:
            test_step(test_images, test_labels,test_loss,test_accuracy)
        print(template.format(epoch + 1,
                              train_loss.result().numpy(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result()*100))

        loss_list.append(train_loss.result().numpy())
        epoch_list.append(epoch+1)

    if args.save_path is not None:
        save_path = osp.expanduser(args.save_path)
        rew_model.save(save_path)
        save_path_agent = '/home/gaoruoyu/PycharmProjects/garage_test/model/ppo_irl'
        ckpt = tf.train.Checkpoint(model=agent_model)
        manager = tf.train.CheckpointManager(ckpt, save_path_agent, max_to_keep=None)
        manager.save()

    plt.plot(epoch_list,loss_list)
    plt.show()

if __name__ == '__main__':
    batch_size = 64
    num_epochs = 300
    rew_model = simple_model()
    loss_object = tf.keras.losses.BinaryCrossentropy()
    lr_fn=tf.keras.optimizers.schedules.ExponentialDecay(0.05,num_epochs, 1e-3,0.1)
    optimizer = tf.keras.optimizers.Adam(lr_fn)
    main(sys.argv)


