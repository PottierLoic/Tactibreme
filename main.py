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


def draw_paws(canvas, paws):
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


def on_canvas_click(event, board: Board, canvas: tk.Canvas):
    """
    Handle clicks on the canvas to determine and print possible moves for the clicked paw.
    """
    col = int(event.x // CELL_SIZE)
    row = int(event.y // CELL_SIZE)
    clicked_paws = board.find_paw_at((row, col))

    if not clicked_paws:
        print(f"No paw found at ({row}, {col})")
        return

    clicked_paws.sort(key=lambda p: p.paw_type.value)
    paw = clicked_paws[0]

    if paw:
        possible_moves = board.possible_movements(paw)
        print(f"Possible moves: {possible_moves}")
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
    board.move_paw(board.paws[5], (4, 0))
    draw_grid(canvas)
    draw_paws(canvas, board.paws)

    canvas.bind("<Button-1>", lambda event: on_canvas_click(event, board, canvas))
    window.mainloop()


if __name__ == "__main__":
    main()
