import numpy as np
import tkinter as tk
import sys

# Constants
N = 32  # Number of rows and columns in the grid
MAX_VALUE = 3  # Maximum value before resetting to 1

# Create a 2D grid to store the values
grid = np.ones((N, N), dtype=int)
grid[0, :] = 2
grid[-1, :] = 2
grid[:, [0, -1]] = 2


# Tkinter event handler for tile click
def on_tile_click(row, col):
    # Increase the value of the clicked tile
    grid[row][col] = (grid[row][col] % MAX_VALUE) + 1
    button_vars[row][col].set(grid[row][col])
    update_button_color(row, col)


# Update button color based on value
def update_button_color(row, col):
    value = grid[row][col]
    color = "white" if value == 1 else "black" if value == 2 else "light blue"
    buttons[row][col].configure(background=color)


# Save grid data to a file and exit the program
def save_and_exit():
    save_grid_data()
    sys.exit()


# Save grid data to a file
def save_grid_data():
    name = name_entry.get().lower().replace(" ", "_")
    filename = f"project/resources/generated_envs/{name}_data.dat"
    with open(filename, "w") as file:
        for row in grid:
            file.write("\t".join(str(value) for value in row))
            file.write("\n")


# Create the GUI window
window = tk.Tk()
window.title("Create Hubert environment")

# Create the grid of buttons
button_vars = [[None] * N for _ in range(N)]
buttons = [[None] * N for _ in range(N)]

for row in range(N):
    for col in range(N):
        # Create a button for each tile
        var = tk.IntVar()
        var.set(grid[row][col])
        button_vars[row][col] = var
        button = tk.Button(window, textvariable=var, width=3, height=1, relief="solid", bd=1)
        button.config(command=lambda r=row, c=col: on_tile_click(r, c))
        button.grid(row=row, column=col, padx=1, pady=1)
        buttons[row][col] = button
        update_button_color(row, col)  # Set initial button color

name_entry = tk.Entry(window)
name_entry.grid(row=N, column=1, columnspan=N-2, pady=10)

# Create the save and exit button
save_button = tk.Button(window, text="Save and Exit", command=save_and_exit)
save_button.grid(row=N+1, column=0, columnspan=N, pady=10)

# Run the GUI event loop
window.mainloop()
