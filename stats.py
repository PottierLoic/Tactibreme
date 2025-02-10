class formatter:
    CYAN = '\033[96m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Stats:
    def __init__(self):
        self.invalid_moves = 0
        self.moves_counter = 0

    def pp_stats(self):
      assert self.moves_counter >= 1
      print(formatter.CYAN)
      print(f"{self.moves_counter} done.")
      print(f"{self.invalid_moves} invalid moves, {(self.invalid_moves / self.moves_counter) * 100}%")
      print(formatter.ENDC)