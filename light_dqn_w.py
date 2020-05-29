import os
import psutil
import light_env as env
import random
import numpy as np
import keras
import time
import matplotlib.pyplot as plt

from collections import deque

class DQN:
    def __init__(self):
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.cloud_model = self.build_cloud_model()
        self.model.load_weights("E:\\vscodeproject\\Python\\simlight\\dqn_10.h5")
        self.cloud_model.load_weights("E:\\vscodeproject\\Python\\simlight\\dqn_night.h5")
        self.update_target_model()
        self.history = {'Episode': [], 'Episode_reward': [], 'Loss': []}

        #if os.path.exists('dqn.h5'):
            #self.model.load_weights('\model\dqn_10.h5')
       
        # 经验池
        self.memory_buffer = deque(maxlen=2000)
        # Q_value的discount rate，以便计算未来reward的折扣回报
        self.gamma = 0.95
        # 贪婪选择法的随机选择行为的程度
        self.epsilon = 1.0
        # 上述参数的衰减率
        self.epsilon_decay = 0.999
        # 最小随机探索的概率
        self.epsilon_min = 0.01
        self.batch = 32

        self.env = env.Light()

    def build_model(self):
        """基本网络结构.
        """
        inputs = keras.layers.Input(shape=(3,))
        x = keras.layers.Dense(4, activation='relu')(inputs)
        x = keras.layers.Dense(4, activation='relu')(x)
        x = keras.layers.Dense(3, activation='linear')(x)

        model = keras.models.Model(inputs=inputs, outputs=x)

        return model

    def build_cloud_model(self):
        """基本网络结构.
        """
        inputs = keras.layers.Input(shape=(3,))
        x = keras.layers.Dense(4, activation='relu')(inputs)
        x = keras.layers.Dense(4, activation='relu')(x)
        x = keras.layers.Dense(3, activation='linear')(x)

        model = keras.models.Model(inputs=inputs, outputs=x)

        return model

    def update_target_model(self):
        """更新target_model
        """
        self.target_model.set_weights(self.model.get_weights())
        #print("target replaced")

    def egreedy_action(self, state):
        """ε-greedy选择action

        Arguments:
            state: 状态
        Returns:
            action: 动作
        """
        choice = ""
        q_values = []
        if np.random.rand() < self.epsilon:
             choice += "rand"
             return random.randint(0, 2), choice, q_values
        else:
            choice+="net "
            q_values = self.model.predict(state)[0]
            return np.argmax(q_values), choice, q_values

    def remember(self, state, action, reward, next_state, done):
        """向经验池添加数据

        Arguments:
            state: 状态
            action: 动作
            reward: 回报
            next_state: 下一个状态
            done: 游戏结束标志
        """
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)

    def update_epsilon(self):
        """更新epsilon
        """
        if self.epsilon >= self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def process_batch(self):
        """batch数据处理
        Arguments:
            batch: batch size

        Returns:
            X: states
            y: [Q_value1, Q_value2]
        """
         # 从经验池中随机采样一个batch
        data = random.sample(self.memory_buffer, self.batch)
        # 生成Q_target。
        states = np.array([d[0] for d in data])
        next_states = np.array([d[3] for d in data])
        y = self.model.predict(states)
        q = self.target_model.predict(next_states)

        for i, (_, action, reward, _, done) in enumerate(data):
            target = reward
            if not done:
                target += self.gamma * np.amax(q[i])
            y[i][action] = target

        return states, y

    def train(self, episode, dis):
        """训练
        Arguments:
            episode: 游戏次数
            batch： batch size
            observation = [self.disnum, lux, envlight, self.power, singlelight]

        Returns:
            history: 训练记录
        """
        tic = time.time()
        self.model.compile(loss='mse', optimizer=keras.optimizers.Adam())
        count = 0
        stop = 0
        for i in range(episode):
            observation = self.env.reset(dis)
            lux = observation[1]
            reward_sum = 0
            loss = np.infty
            done = False
            x = self.process_observation(observation)

            while not done and i < 10:
                output_trained = self.cloud_model.predict(x)[0]
                action_cloud = np.argmax(output_trained)
                observation, reward, done, _ = self.env.step(dis, action_cloud)
                observation = self.process_observation(observation)
                #reward_sum += reward
                self.remember(x[0], action_cloud, reward, observation[0], done)
                if len(self.memory_buffer) > self.batch:
                    X, y = self.process_batch()
                    loss = self.model.train_on_batch(X, y)
                    count += 1
                x = observation
            done = False
            #training_model_reward_sum = 0
            observation = self.env.reset(dis)

            x = self.process_observation(observation)
            while not done:
                # 通过贪婪选择法ε-greedy选择action。
                action, choice, q_values = self.egreedy_action(x)
                observation, reward, done, info = self.env.step(dis, action)

                lux = observation[1]
                observation = self.process_observation(observation)
                # 将数据加入到经验池。
                # print(i, "action:", action,"epsilon {:.3f}".format(self.epsilon),"choice:", choice, 
                #         "observation:", observation, "reward {:.4f}".format(reward),"loss {:.4f}".format(loss)
                #         ,"q_values:", q_values)
                reward_sum += reward
                self.remember(x[0], action, reward, observation[0], done)

                if len(self.memory_buffer) > self.batch:
                    # 训练
                    X, y = self.process_batch()
                    loss = self.model.train_on_batch(X, y)
                    count += 1
                    # 减小egreedy的epsilon参数。
                    self.update_epsilon()
                    # 固定次数更新target_model
                    if count != 0 and count % 20 == 0:
                        self.update_target_model()
                #time.sleep(0.1)
                x = observation

            if reward_sum > 450:
                stop += 1
                if stop > 4:
                    toc = time.time()
                    print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
                    print("总耗时", toc - tic)
                    os._exit()
            else:
                stop = 0

            if i % 1 == 0:
                self.history['Episode'].append(i)
                self.history['Episode_reward'].append(reward_sum)
                self.history['Loss'].append(loss)
                print('Episode: {} | Episode reward: {} | loss: {:.3f} | e:{:.2f}'.format(i, reward_sum, loss, self.epsilon))


        #if dis == 'dis10':
            #self.model.save_weights('dqn.h5')
        print(u'当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )

        self.plot()
        # return history

    def process_observation(self, state):
        state = np.array([state[0], state[2], state[4]])
        return state.reshape(-1, 3)

    def plot(self):
        '''
        -600,600
        0,10
        '''
        x = self.history['Episode']
        r = self.history['Episode_reward']
        l = self.history['Loss']
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121)
        plt.ylim(-400, 600)
        ax.plot(x, r)
        ax.set_title('Episode_reward')
        ax.set_xlabel('episode')
        ax = fig.add_subplot(122)
        plt.ylim(0, 4)
        ax.plot(x, l)
        ax.set_title('Loss')
        ax.set_xlabel('episode')
        plt.show()

if __name__ == '__main__':
    lightDQN = DQN()
    string = "dis"
    for i in range(1, 11):
        lightDQN.train(50, string + str(i))


