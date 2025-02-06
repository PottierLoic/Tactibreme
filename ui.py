import matplotlib.pyplot as plt
import threading
import time

def plot_rewards(game):
    plt.ion()
    fig, ax = plt.subplots()

    rewards_agent = []

    while True:
        rewards_agent = [entry[2] for entry in game.agent.memory]

        ax.clear()
        ax.plot(rewards_agent, label='Agent 1 Rewards', color='blue')
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
