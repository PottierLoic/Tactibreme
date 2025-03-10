import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import pygame
import argparse
import threading
import pygame
from board import GameFinished
from constants import *
from game import Game
from logger import get_logger
from paw import Paw
from color import Color
import sys

lock = threading.Lock()
STOP_EVENT = threading.Event()
pool = []

def parse_args():
    parser = argparse.ArgumentParser(
        description="TACTIBREME: Train an AI and play board game matches"
    )
    parser.add_argument("--ui", action="store_true", help="Enable pygame UI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train Command
    train_parser = subparsers.add_parser("train", help="Train an AI model")
    train_parser.add_argument("name", type=str, help="Name of the AI model")
    train_parser.add_argument("--count", type=int, default=1000, help="Number of games to train")
    train_parser.add_argument("--load", nargs=2, metavar=("AGENT1", "AGENT2"), help="Paths to two models")
    train_parser.add_argument("--epsilon", type=float, help="Exploration rate")
    train_parser.add_argument("--decay", type=float, help="Decay rate")
    train_parser.add_argument("--gamma", type=float, help="Discount factor")
    train_parser.add_argument("--lr", type=float, help="Learning rate")
    train_parser.add_argument("--record", type=str, help="File to save match history")
    train_parser.add_argument("--stats", type=str, help="File to save training statistics")
    train_parser.add_argument("--ui", action="store_true", help="Enable pygame UI.")

    # AI vs AI Command
    record_parser = subparsers.add_parser("record", help="Play AI vs AI matches")
    record_parser.add_argument("--count", type=int, default=100, help="Number of matches to play")
    record_parser.add_argument("--blue", type=str, required=True, help="Path to blue AI model")
    record_parser.add_argument("--red", type=str, required=True, help="Path to red AI model")
    record_parser.add_argument("--ui", action="store_true", help="Enable pygame UI")

    # Human vs AI Command
    play_parser = subparsers.add_parser("play", help="Play a match against AI")
    play_parser.add_argument("--ai", type=str, required=True, help="Path to AI model")
    return parser.parse_args()


class Context:
    def __init__(self) -> None:
        self.game = Game()
        self.selected_paw = None
        self.possible_moves = []

    def draw_grid(self, screen: pygame.Surface) -> None:
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                color = WHITE if (row + col) % 2 == 0 else BLACK
                pygame.draw.rect(
                    screen,
                    color,
                    pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                )

    def draw_paws(self, screen: pygame.Surface) -> None:
        with lock:
            paws_copy = dict(self.game.board.paws_coverage)
        for position, paws in paws_copy.items():
            sorted_paws = sorted(paws, key=lambda paw: paw.paw_type.value)
            for paw in sorted_paws:
                row, col = position
                x = col * CELL_SIZE + CELL_SIZE // 2
                y = row * CELL_SIZE + CELL_SIZE // 2
                radius = CELL_SIZE // (3 + (paw.paw_type.value / 2))
                pygame.draw.circle(
                    screen, BLACK, (int(x), int(y)), int(radius + 2)
                )
                pygame.draw.circle(
                    screen, paw.color.name, (int(x), int(y)), int(radius)
                )
                font = pygame.font.Font(None, 24)
                text = font.render(str(paw.paw_type.value), True, WHITE)
                text_rect = text.get_rect(center=(x, y))
                screen.blit(text, text_rect)

    def highlight_moves(self, screen: pygame.Surface) -> None:
        with lock:
            moves_copy = list(self.possible_moves)
        for row, col in moves_copy:
            highlight_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            highlight_surface.fill((255, 255, 0, 128))
            screen.blit(highlight_surface, (col * CELL_SIZE, row * CELL_SIZE))

    def handle_left_click(self, pos: tuple[int, int]) -> None:
        col = pos[0] // CELL_SIZE
        row = pos[1] // CELL_SIZE
        clicked_paws = self.game.board.find_paw_at((row, col))
        if not clicked_paws:
            self.selected_paw = None
            self.possible_moves = []
            return

        filtered_paws = [
            paw for paw in clicked_paws if paw.color == self.game.current_turn
        ]
        self.selected_paw = filtered_paws[0] if filtered_paws else None
        if self.selected_paw:
            self.possible_moves = self.game.board.possible_movements(self.selected_paw)
        else:
            self.possible_moves = []

    def handle_right_click(self, pos: tuple[int, int]) -> None:
        if not self.selected_paw:
            return
        col = pos[0] // CELL_SIZE
        row = pos[1] // CELL_SIZE
        destination = (row, col)
        if destination in self.possible_moves:
            try:
                result = self.game.play_turn(self.selected_paw, destination)
                self.selected_paw = None
                self.possible_moves = []
                if result == 1:
                    raise GameFinished(winner_color=self.game.current_turn)
            except GameFinished as game_ended:
                get_logger(__name__).debug(f"The winner is {game_ended.winner_color} !!!!")
                return

    def handle_arrow_key(self) -> None:
        if not self.selected_paw:
            return
        unicolor_list = self.game.board.get_unicolor_list(
            self.game.board.paws_coverage[self.selected_paw.position],
            self.game.current_turn,
        )
        if len(unicolor_list) > 1:
            current_index = unicolor_list.index(self.selected_paw)
            next_index = (current_index + 1) % len(unicolor_list)
            self.selected_paw = unicolor_list[next_index]
            self.possible_moves = self.game.board.possible_movements(self.selected_paw)


def run_training(args, controller):
    """Runs the training in a separate thread while UI runs on main thread"""
    if args.load:
        game = Game(
            agent1_path=args.load[0],
            agent2_path=args.load[1],
            num_games=args.count,
            mode="train",
            model_name=args.name,
        )
        get_logger(__name__).info(
            f"Continuing training from {args.load[0]} and {args.load[1]}"
        )
    else:
        valid_hyperparams = {
            k: v
            for k, v in vars(args).items()
            if v is not None and k in ["epsilon", "decay", "gamma", "lr"]
        }
        game = Game(
            num_games=args.count,
            mode="train",
            model_name=args.name,
            **valid_hyperparams
        )
    with lock:
        controller.game = game
    game.train_agents(STOP_EVENT)
    STOP_EVENT.set()

def run_ui(args, controller):
    """Runs the pygame UI"""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Les Tacticiens de Brême")

    if args.command == "train":
        task_thread = threading.Thread(target=run_training, args=(args, controller))
        task_thread.start()
        pool.append(task_thread)
    elif args.command == "record":
        task_thread = threading.Thread(target=run_record, args=(args, controller))
        task_thread.start()
        pool.append(task_thread)

    while not STOP_EVENT.is_set():
        screen.fill(BLACK)
        controller.draw_grid(screen)
        controller.draw_paws(screen)
        if controller.selected_paw:
            controller.highlight_moves(screen)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                STOP_EVENT.set()

    pygame.quit()

def run_human_vs_ai(args, controller):
    """Runs Human vs AI game on the main thread with UI"""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Les Tacticiens de Brême - Humain vs AI")

    ai_color = "RED"
    game = Game(
        agent1_path=args.ai,
        num_games=1,
        mode="human_vs_ai",
    )
    with lock:
        controller.game = game

    fake_stop_event = threading.Event()
    controller.game.draft(fake_stop_event)

    while not STOP_EVENT.is_set():
        screen.fill(BLACK)
        controller.draw_grid(screen)
        controller.draw_paws(screen)
        if controller.selected_paw:
            controller.highlight_moves(screen)

        with lock:
            if not STOP_EVENT.is_set():
                try:
                    controller.game.play_ai_turn()
                except GameFinished as game_ended:
                    get_logger(__name__).debug(
                        f"The winner is {game_ended.winner_color} !!!!"
                    )
                    STOP_EVENT.set()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                STOP_EVENT.set()
            elif event.type == pygame.MOUSEBUTTONDOWN and controller.game.current_turn == Color.BLUE:
                if event.button == 1:
                    controller.handle_left_click(event.pos)
                elif event.button == 3:
                    controller.handle_right_click(event.pos)
            elif event.type == pygame.KEYDOWN and controller.game.current_turn == Color.BLUE:
                if event.key == pygame.K_RIGHT:
                    controller.handle_arrow_key()

        pygame.display.flip()
    pygame.quit()

def run_record(args, controller):
    """Runs AI vs AI matches while UI runs on main thread"""
    game = Game(
        agent1_path=args.blue,
        agent2_path=args.red,
        num_games=args.count,
        model_name="",
        mode="ai_vs_ai"
    )
    get_logger(__name__).info(
        f"Starting AI vs AI matches with {args.blue} (blue) vs {args.red} (red)"
    )
    with lock:
        controller.game = game
    game.record_games(STOP_EVENT)

def main():
    args = parse_args()
    controller = Context()
    command_map = {
        "train": run_training,
        "record": run_record,
        "play": run_human_vs_ai
    }
    if args.command == "play" or args.ui:
        command_map[args.command](args, controller)
    else:
        task_thread = threading.Thread(target=command_map[args.command], args=(args, controller))
        task_thread.start()
        pool.append(task_thread)
        task_thread.join()
    STOP_EVENT.set()
    for thread in pool:
        thread.join()
    sys.exit()

if __name__ == "__main__":
    main()
