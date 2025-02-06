import pygame
from logger import get_logger
from board import GameFinished
from constants import *
from game import Game
from paw import Paw
from ui import *


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
        for position, paws in self.game.board.paws_coverage.items():
            sorted_paws = sorted(paws, key=lambda paw: paw.paw_type.value)
            for paw in sorted_paws:
                row, col = position
                x = col * CELL_SIZE + CELL_SIZE // 2
                y = row * CELL_SIZE + CELL_SIZE // 2
                radius = CELL_SIZE // (3 + (paw.paw_type.value / 2))
                pygame.draw.circle(
                    screen, paw.color.name, (int(x), int(y)), int(radius)
                )
                font = pygame.font.Font(None, 24)
                text = font.render(str(paw.paw_type.value), True, WHITE)
                text_rect = text.get_rect(center=(x, y))
                screen.blit(text, text_rect)

    def highlight_moves(self, screen: pygame.Surface) -> None:
        for row, col in self.possible_moves:
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
                message = self.game.play_turn(self.selected_paw, destination)
                self.selected_paw = None
                self.possible_moves = []
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

def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Les Tacticiens de Brême")
    controller = Context()
    start_plotting_thread(controller.game)
    running = True
    while running:
        screen.fill(BLACK)
        controller.draw_grid(screen)
        controller.draw_paws(screen)
        if controller.selected_paw:
            controller.highlight_moves(screen)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    controller.handle_left_click(event.pos)
                elif event.button == 3:
                    controller.handle_right_click(event.pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    controller.handle_arrow_key()
                elif event.key == pygame.K_DOWN:
                    controller.handle_arrow_key()
                elif event.key == pygame.K_SPACE:
                    controller.game.real_player = not controller.game.real_player
        if not controller.game.real_player:
            controller.game.play_turn()
    pygame.quit()


if __name__ == "__main__":
    main()
