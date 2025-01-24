import matplotlib.pyplot as plt
import threading
import time

def plot_rewards(game):
    plt.ion()
    fig, ax = plt.subplots()

    rewards_agent1 = []
    rewards_agent2 = []

    while True:
        rewards_agent1 = [entry[2] for entry in game.agent1.memory]
        rewards_agent2 = [entry[2] for entry in game.agent2.memory]

        ax.clear()
        ax.plot(rewards_agent1, label='Agent 1 Rewards', color='blue')
        ax.plot(rewards_agent2, label='Agent 2 Rewards', color='red')
        ax.set_title('AI rewards over episodes')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.legend()

        plt.pause(1)

    plt.ioff()
    plt.show()

def start_plotting_thread(game):
    plot_thread = threading.Thread(target=plot_rewards, args=(game,), daemon=True)
    plot_thread.start()
