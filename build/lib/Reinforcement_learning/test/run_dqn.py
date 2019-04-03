from Reinforcement_learning import DQN
from Reinforcement_learning import Maze
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
times = []

def update():
    step = 0
    for episode in range(100):
        temp_times = 0
        # initial observation
        observation = env.reset()

        while True:
            temp_times += 1
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                times.append(temp_times)
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DQN(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      # output_graph=True
                      )
    env.after(100, update)
    env.mainloop()
    plt.plot(np.arange(100), times)
    plt.savefig('./result.jpg')
    print(np.array(times).mean())
    print(np.array(times[50:]).mean())