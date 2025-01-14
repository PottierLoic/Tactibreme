import sys
import tkinter as tk

from board import Board
from paw import Paw, PawType

SCREEN_SIZE = 500
GRID_SIZE = 5
CELL_SIZE = SCREEN_SIZE / GRID_SIZE


def draw_grid(canvas):
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            x1 = col * CELL_SIZE
            y1 = row * CELL_SIZE
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE
            color = "white" if (row + col) % 2 == 0 else "black"
            canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")


def draw_paws(canvas: tk.Canvas, paws_coverage: dict[tuple[int, int], list[Paw]]):
    for position, paws in paws_coverage.items():
        for paw in paws:
            row, col = paw.position
            x = col * CELL_SIZE + CELL_SIZE / 2
            y = row * CELL_SIZE + CELL_SIZE / 2
            radius = CELL_SIZE / (3 + (paw.paw_type.value / 2))
            canvas.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=paw.color,
                outline="black",
            )
            canvas.create_text(
                x, y, text=paw.paw_type.value, fill="white", font=("Arial", 16, "bold")
            )
    print(str(paws_coverage))


def on_canvas_left_click(event, board: Board, canvas: tk.Canvas):
    global selected_position, selected_stack, selected_stack_index
    
    col = int(event.x // CELL_SIZE)
    row = int(event.y // CELL_SIZE)
    clicked_paws = board.find_paw_at((row, col))

    if not clicked_paws:
        print(f"No paw found at ({row}, {col})")
        selected_position = None
        selected_stack = []
        selected_stack_index = 0
        canvas.delete("highlight")
        return

    clicked_paws.sort(key=lambda p: p.paw_type.value)
    
    selected_position = (row, col)
    selected_stack = clicked_paws
    selected_stack_index = 0
    
    selected_paw = selected_stack[selected_stack_index]
    possible_moves = board.possible_movements(selected_paw)
    highlight_moves(canvas, possible_moves)

def on_canvas_right_click(event, board: Board, canvas: tk.Canvas):
    global selected_position, selected_stack, selected_stack_index

    if not selected_stack:
        print("No paw selected to move.")
        return

    col = int(event.x // CELL_SIZE)
    row = int(event.y // CELL_SIZE)
    destination = (row, col)

    selected_paw = selected_stack[selected_stack_index]

    possible_moves = board.possible_movements(selected_paw)
    if destination not in possible_moves:
        print(f"Destination {destination} is not a valid move for {selected_paw.paw_type}.")
        return

    try:
        board.move_paw(selected_paw, destination)
    except ValueError as e:
        print(f"Move failed: {e}")
        return

    canvas.delete("all")
    draw_grid(canvas)
    draw_paws(canvas, board.paws_coverage)

    winner = board.check_win()
    if winner != "none":
        print(f"We have a winner: {winner}!")

    selected_position = None
    selected_stack = []
    selected_stack_index = 0

def on_arrow_key(event, board: Board, canvas: tk.Canvas, direction: int):
    global selected_position, selected_stack, selected_stack_index

    if not selected_stack or len(selected_stack) < 2:
        return

    selected_stack_index = (selected_stack_index + direction) % len(selected_stack)

    selected_paw = selected_stack[selected_stack_index]
    possible_moves = board.possible_movements(selected_paw)
    print(f"Selected {selected_paw.paw_type} at {selected_position}, possible moves: {possible_moves}")

    canvas.delete("all")
    draw_grid(canvas)
    draw_paws(canvas, board.paws_coverage)
    highlight_moves(canvas, possible_moves)

def highlight_moves(canvas: tk.Canvas, moves: list[tuple[int, int]]):
    """
    Highlight possible moves on the canvas.
    """
    canvas.delete("highlight")
    for row, col in moves:
        x1 = col * CELL_SIZE
        y1 = row * CELL_SIZE
        x2 = x1 + CELL_SIZE
        y2 = y1 + CELL_SIZE
        canvas.create_rectangle(
            x1, y1, x2, y2, width=4, outline="yellow", stipple="gray50", tag="highlight"
        )


def main():
    args = sys.argv
    window = tk.Tk()
    window.config(bg="black")
    window.title("Les Tacticiens de Brême")
    window.geometry("500x500")

    canvas = tk.Canvas(window, width=SCREEN_SIZE, height=SCREEN_SIZE)
    canvas.pack()

    board = Board()
    draw_grid(canvas)
    draw_paws(canvas, board.paws_coverage)

    canvas.bind("<Button-1>", lambda event: on_canvas_left_click(event, board, canvas))
    canvas.bind("<Button-3>", lambda event: on_canvas_right_click(event, board, canvas))
    window.bind("<Up>", lambda event: on_arrow_key(event, board, canvas, direction=-1))
    window.bind("<Down>", lambda event: on_arrow_key(event, board, canvas, direction=1))
    window.mainloop()


if __name__ == "__main__":
    main()
