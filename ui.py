import matplotlib.pyplot as plt
import threading
import time

class Coord:
    def __init__(self, value, label):
        self.value = value
        self.label = label

class Plot:
    def __init__(self, values, label, color):
        self.values = values
        self.label = label
        self.color = color

def plot_rewards(game):
    plt.ion()
    fig, axes = plt.subplots(2, 2)

    def plot(title: str, x: Coord, y: Coord, plots: [Plot]):
        axes[x.value, y.value].clear()
        for plot in plots:
            axes[x.value, y.value].plot(plot.values, label=plot.label, color=plot.color)
        axes[x.value, y.value].set_title(title)
        axes[x.value, y.value].set_xlabel(x.label)
        axes[x.value, y.value].set_ylabel(y.label)
        axes[x.value, y.value].legend()

    rewards_agent1 = []
    rewards_agent2 = []
    rewards_counter_blue = []
    rewards_counter_red  = []

    while True:
        rewards_agent1 = [entry[2] for entry in game.agent1.memory]
        rewards_agent2 = [entry[2] for entry in game.agent2.memory]
        plot(
            title="AI rewards over episodes",
            x=Coord(0, "Episode"),
            y=Coord(0, "Reward"),
            plots=[
                Plot(rewards_agent1,"Agent Blue Rewards", "blue"),
                Plot(rewards_agent2,"Agent Red Rewards", "red")
            ]
        )

        rewards_counter_blue.append(game.agent1.reward_counter)
        plot(
            title="Blue Agent reward counter over episodes",
            x=Coord(1, "Episode"),
            y=Coord(0, "Reward counter"),
            plots=[
                Plot(rewards_counter_blue,"Blue Agent Rewards", "blue")
            ]
        )


        rewards_counter_red.append(game.agent2.reward_counter)
        plot(
            title="Red Agent reward counter over episodes",
            x=Coord(1, "Episode"),
            y=Coord(1, "Reward counter"),
            plots=[
                Plot(rewards_counter_red,"Red Agent Rewards", "red")
            ]
        )

        plt.pause(1)

    plt.ioff()
    plt.show()

def start_plotting_thread(game):
    plot_thread = threading.Thread(target=plot_rewards, args=(game,), daemon=True)
    plot_thread.start()
