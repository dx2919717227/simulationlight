import pymysql
import numpy as np
import time

class Light(object):
    def __init__(self):
        '''
        self.positive_reward:台灯不需要调整，获得正奖励
        self.negative_reward:台灯被调整，获得负奖励
        self.positive_reward_decay:奖励衰减率
        self.user_operation:0为未操作台灯 1为操作台灯
        self.power:台灯功率
        '''
        self.max_reward = 1
        self.positive_reward = 1   # 用户未调整灯
        self.negative_reward = -1 # 用户对灯进行了调整
        self.positive_reward_decay = 0.999
        self.disnum = 0.028
        self.power = 350
        self.max_step = 600
        self.count_step = 0
        self.lux = 328
        self.db = pymysql.connect("localhost", port=3306, user="root",password="123456", db="light")
        self.cusor = self.db.cursor()

    def get_state(self, dis):
        sql1 = "select %s from lightdata where power='%s'" % (dis, self.power)
        sql2 = "select %s from lightenv where id='1'" % dis
        sql3 = "select %s from lightsingle where power='%s'" % (dis, self.power)
        self.cusor.execute(sql1)
        lux = self.cusor.fetchone()
        self.cusor.execute(sql2)
        envlight = self.cusor.fetchone()
        self.cusor.execute(sql3)
        singlelight = self.cusor.fetchone()
        return lux[0], envlight[0], singlelight[0]

    def step(self, dis, action):
        '''
        action:0保持不变，1 power-10，2 power +10
        '''
        self.disnum = float(dis[3:]) * 0.028
        done = True if self.count_step == self.max_step else False
        self.count_step += 1
        if action == 0:
            lux, envlight, singlelight = self.get_state(dis)
        elif action == 1:
            if self.power != 250:
                self.power -= 10
            lux, envlight, singlelight = self.get_state(dis)
        else:
            if self.power != 450:
                self.power += 10
            lux, envlight, singlelight = self.get_state(dis)

        self.lux = lux
        reward = self.get_reward(action)
        info = {
            "power":self.power,
            "dis":self.disnum,
            "lux":lux,
            "envlight":envlight,
            "singlelight":singlelight,
            "action":action,
            "reward":reward,
            "done":done
        }
        
        state = [self.disnum, lux, envlight, self.power, singlelight]
        return state, reward, done, info

    def get_reward(self, action):
        #310 330 reward 1 -10 self.gamma=0.5 nice
        if self.lux >= 350 and self.lux <= 380:
            #if action == 0:
            if action == 0:
                return 1
            else:
                reward = self.positive_reward
                self.positive_reward *= self.positive_reward_decay
                return reward
            # else:
            #     self.positive_reward = self.max_reward
            #     return self.negative_reward
        if self.lux < 350:
            if action == 2:
                reward = self.positive_reward
                self.positive_reward *= self.positive_reward_decay
                return reward
            else:
                self.positive_reward = self.max_reward
                return self.negative_reward

        if self.lux > 380:
            if action == 1:
                reward = self.positive_reward
                self.positive_reward *= self.positive_reward_decay
                return reward
            else:
                self.positive_reward = self.max_reward
                return self.negative_reward
        # if self.user_operation == 1:    #用户调节灯 返回奖励-10
        #     self.positive_reward = 10
        #     return self.negative_reward
        # else:
        #     reward = self.positive_reward
        #     self.positive_reward *= self.positive_reward_decay
        #     return reward
        

    def reset(self, dis):
        self.db.close()
        self.__init__()
        self.disnum = float(dis[3:]) * 0.028
        lux, envlight, singlelight = self.get_state(dis)
        state = [self.disnum, lux, envlight, self.power, singlelight]
        return state


if __name__ == "__main__":
    light = Light()
    dis = 'dis10'
    for i in range(50):
        done = False
        observation = light.reset(dis)
        print(observation)
        while not done:
            action = np.random.randint(0, 3)
            state, reward, done ,info = light.step(dis, action)
            print("info:", info)
            time.sleep(1)

            


