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
        radius = CELL_SIZE / 4
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


def main():
    window = tk.Tk()
    window.config(bg="black")
    window.title("Les Tacticiens de Brême")
    window.geometry("500x500")

    canvas = tk.Canvas(window, width=SCREEN_SIZE, height=SCREEN_SIZE)
    canvas.pack()

    board = Board()
    draw_grid(canvas)
    draw_paws(canvas, board.paws)

    window.mainloop()


if __name__ == "__main__":
    main()
