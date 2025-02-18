import csv

def init_writer(filename):
    csvfile = open(filename, 'w', newline='')
    writer = csv.writer(csvfile, delimiter=',',
                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['agent', 'color', 'id_paw', 'dest_x', 'dest_y', 'reward', 'win', 'epsilon'])
    return writer, csvfile

class WriterBuffer:
    def __init__(self, writer):
        self.agent = None
        self.color = None
        self.id_paw = None
        self.dest_x = None
        self.dest_y = None
        self.reward = None
        self.win = None
        self.epsilon = None
        self.writer = writer

    def reset_line(self):
        self.agent = None
        self.color = None
        self.id_paw = None
        self.dest_x = None
        self.dest_y = None
        self.reward = None
        self.win = None
        self.epsilon = None

    def push(self):
        self.writer.writerow([self.agent, self.color, self.id_paw, self.dest_x, self.dest_y, self.reward, self.win, self.epsilon])

    def set_agent(self, agent):
        self.agent = agent

    def set_color(self, color):
        self.color = color

    def set_paw(self, paw):
        self.id_paw = paw

    def set_dest(self, x, y):
        self.dest_x = x
        self.dest_y = y

    def set_reward(self, reward):
        self.reward = reward

    def set_win(self, win):
        self.win = win

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon