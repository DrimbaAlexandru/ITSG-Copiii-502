import tkinter as tk

from gui.gui import AppGUI


class App:
    def __init__(self):
        root = tk.Tk()
        AppGUI(root)
        root.mainloop()
        root.destroy()

App()