import pygame
from board import Board
from paw import Paw, PawType
from constants import *

class Context:
    def __init__(self):
        self.board = Board()
        self.selected_stack = []
        self.selected_stack_index = 0

    def draw_grid(self, screen):
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                color = WHITE if (row + col) % 2 == 0 else BLACK
                pygame.draw.rect(
                    screen, color, pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )

    def draw_paws(self, screen):
        for position, paws in self.board.paws_coverage.items():
            sorted_paws = sorted(paws, key=lambda paw: paw.paw_type.value) # TODO: remove when "entonnoir" method is done :)"
            for paw in sorted_paws:
                row, col = position
                x = col * CELL_SIZE + CELL_SIZE // 2
                y = row * CELL_SIZE + CELL_SIZE // 2
                radius = CELL_SIZE // (3 + (paw.paw_type.value / 2))
                pygame.draw.circle(screen, paw.color, (int(x), int(y)), int(radius))
                font = pygame.font.Font(None, 24)
                text = font.render(str(paw.paw_type.value), True, WHITE)
                text_rect = text.get_rect(center=(x, y))
                screen.blit(text, text_rect)

    def highlight_moves(self, screen, moves):
        for row, col in moves:
            highlight_surface = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
            highlight_surface.fill((255, 255, 0, 128))
            screen.blit(highlight_surface, (col * CELL_SIZE, row * CELL_SIZE))

    def handle_left_click(self, pos):
        col = pos[0] // CELL_SIZE
        row = pos[1] // CELL_SIZE
        clicked_paws = self.board.find_paw_at((row, col))

        if not clicked_paws:
            self.selected_stack = []
            self.selected_stack_index = 0
            return

        self.selected_stack = clicked_paws
        self.selected_stack_index = 0

    def handle_right_click(self, screen, pos):
        if not self.selected_stack:
            return

        col = pos[0] // CELL_SIZE
        row = pos[1] // CELL_SIZE
        destination = (row, col)

        selected_paw = self.selected_stack[self.selected_stack_index]

        possible_moves = self.board.possible_movements(selected_paw)
        if destination not in possible_moves:
            return

        try:
            self.board.move_paw(selected_paw, destination)
        except ValueError:
            return

    def handle_arrow_key(self, direction):
        if not self.selected_stack or len(self.selected_stack) < 2:
            return

        self.selected_stack_index = (self.selected_stack_index + direction) % len(self.selected_stack)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    pygame.display.set_caption("Les Tacticiens de Brême")
    controller = Context()

    running = True
    while running:
        screen.fill(BLACK)
        controller.draw_grid(screen)
        controller.draw_paws(screen)

        if controller.selected_stack:
            selected_paw = controller.selected_stack[controller.selected_stack_index]
            possible_moves = controller.board.possible_movements(selected_paw)
            controller.highlight_moves(screen, possible_moves)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    controller.handle_left_click(event.pos)
                elif event.button == 3:
                    controller.handle_right_click(screen, event.pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    controller.handle_arrow_key(-1)
                elif event.key == pygame.K_DOWN:
                    controller.handle_arrow_key(1)

    pygame.quit()

if __name__ == "__main__":
    main()
