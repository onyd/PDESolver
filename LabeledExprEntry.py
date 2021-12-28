from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sympy as sp


class LabeledExprEntry(Frame):
    def __init__(self, master) -> None:
        Frame.__init__(self, master, bg="#e1e1e1")

        self.fig = Figure(figsize=(0.5, 0.5), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side="left", fill="both", expand=True)

        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)

        self.entry = Entry(self, width=10)
        self.entry.insert(0, "0")
        self.entry.pack(side="left", fill="both", expand=True)

    def get_expr(self):
        return sp.parse_expr(self.entry.get(), evaluate=False)

    def set_text(self, text):
        tmptext = "$" + text + "$"

        self.ax.clear()
        self.ax.text(0.5, 0.5, tmptext, fontsize=12, ha="center", va="center")
        self.canvas.draw()