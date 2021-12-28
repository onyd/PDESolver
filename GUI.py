"""
author: Anthony Dard
description: this script is a tkinter GUI PDE solver which allows user to:

To add algorithm, see the __init__ doc of PDEWindow
"""

from tkinter import *
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from collections import deque
from sympy import lambdify
from sympy.abc import x, y, t

from LabeledExprEntry import LabeledExprEntry


class PDEWindow(Tk):
    def __init__(self, compute_algorithms) -> None:
        """compute_algorithms is a list of dict {"name" : string, "algo" : function) where:
        -string is the text in button selection of algorithm
        -function has the signature: def <name>(second_derivatives_coefs, first_derivatives_coefs, function_coef, right_side_function, *parameters, resolutions, [t], bounds) -> 2D array of the solution function
        """
        super().__init__()

        self.title("PDE solver")
        self.resizable(False, False)

        self.compute_algorithms = compute_algorithms

        self.coefficient_entries = []
        self.condition_entries = []

        self.setup_plots()
        self.setup_panel()

    def setup_panel(self):
        # Right panel for options
        self.frame_pannel = Frame(self, relief=RAISED, bg="#e1e1e1")
        self.frame_PDE_type = Frame(self.frame_pannel, bg="#e1e1e1")
        self.frame_PDE_equation = Frame(self.frame_pannel, bg="#e1e1e1")
        self.frame_PDE_coefficients = Frame(self.frame_pannel, bg="#e1e1e1")
        self.frame_conditions_expr = Frame(self.frame_pannel, bg="#e1e1e1")
        self.frame_resolution_sliders = Frame(self.frame_pannel, bg="#e1e1e1")
        self.frame_parameters = Frame(self.frame_pannel, bg="#e1e1e1")
        self.frame_buttons = Frame(self.frame_pannel, bg="#e1e1e1")
        self.frame_action_buttons = Frame(self.frame_pannel, bg="#e1e1e1")

        # Parameters
        self.label_parameters = []
        self.parameters = []

        # Equation display
        self.PDE_fig = Figure(figsize=(3, 1), dpi=100)
        self.PDE_ax = self.PDE_fig.add_subplot(111)

        self.PDE_canvas = FigureCanvasTkAgg(self.PDE_fig,
                                            master=self.frame_PDE_equation)
        self.PDE_canvas.get_tk_widget().pack(side="top",
                                             fill="both",
                                             expand=True)

        self.PDE_ax.get_xaxis().set_visible(False)
        self.PDE_ax.get_yaxis().set_visible(False)

        # Selection of PDE type
        self.label_PDE_type = Label(self.frame_PDE_type, text="PDE types:")
        self.label_PDE_type.grid(row=0, column=1)
        PDE_types = [algo["name"] for algo in self.compute_algorithms]

        self.check_PDE_buttons = [None] * len(self.compute_algorithms)
        self.PDE_type = IntVar()
        self.PDE_type.set(0)
        self.handle_algo_selection()

        for i in range(len(self.compute_algorithms)):
            self.check_PDE_buttons[i] = Radiobutton(self.frame_PDE_type,
                                                    value=i,
                                                    text=PDE_types[i],
                                                    variable=self.PDE_type)
            self.check_PDE_buttons[i].grid(row=i // 3 + 1, column=i % 3)
            self.check_PDE_buttons[i].bind(
                "<ButtonRelease-1>",
                lambda event: self.after(100, self.handle_algo_selection),
            )

        # Slider for parameter update
        self.label_resolution = Label(self.frame_resolution_sliders,
                                      text="Resolution: ")
        self.slider_resolution = Scale(
            self.frame_resolution_sliders,
            from_=5,
            to=500,
            orient=HORIZONTAL,
            bg="#e1e1e1",
        )
        self.slider_resolution.set(50)
        self.label_resolution.pack(side=LEFT)
        self.slider_resolution.pack(fill="x")

        # Frame pack
        self.frame_pannel.grid(row=1,
                               column=0,
                               padx=2,
                               pady=2,
                               columnspan=2,
                               sticky="nswe")
        self.frame_PDE_type.pack(fill="x")
        self.frame_PDE_equation.pack(fill="x")
        self.frame_PDE_coefficients.pack(fill="x")
        self.frame_conditions_expr.pack(fill="x")
        self.frame_resolution_sliders.pack(fill="x")

        self.button_start = Button(self.frame_action_buttons, text="Start")
        self.button_start.pack(side=TOP, fill="x")
        self.button_start.bind("<ButtonRelease-1>", lambda event: self.start())

        self.button_stop = Button(self.frame_action_buttons, text="Stop")
        self.button_stop.pack(side=TOP, fill="x")
        self.button_stop.bind("<ButtonRelease-1>", lambda event: self.stop())

        self.button_reset = Button(self.frame_action_buttons, text="Reset")
        self.button_reset.pack(side=TOP, fill="x")
        self.button_reset.bind("<ButtonRelease-1>", lambda event: self.reset())

        self.frame_action_buttons.pack(fill="x")

    def setup_plots(self):
        self.frame_plots = Frame(self, bg="#e1e1e1")

        self.fig = Figure(figsize=(10, 5), dpi=100)
        self.ax_2D = self.fig.add_subplot(121)
        self.ax_3D = self.fig.add_subplot(122, projection="3d")

        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.plot_canvas.get_tk_widget().grid(row=0,
                                              column=0,
                                              padx=2,
                                              pady=2,
                                              sticky="nswe")
        self.frame_plots.grid(row=0, column=0, padx=2, pady=2, sticky="nswe")

    def set_PDE_equation(self, latex_equation):
        tmptext = "$" + latex_equation + "$"

        self.PDE_ax.clear()
        self.PDE_ax.text(0.5,
                         0.5,
                         tmptext,
                         fontsize=16,
                         ha="center",
                         va="center")
        self.PDE_canvas.draw()

    # Drawing
    def get_parameters(self):
        result = []
        for parameter, parameter_options, parameter_widget in self.parameters:
            if parameter_options["type"] == "slider":
                result.append(parameter_widget.get())
            elif parameter_options["type"] == "check":
                result.append(parameter_widget.instate(["selected"]))
            elif parameter_options["type"] == "expression":
                result.append(parameter_widget.get_expr())

        return result

    def get_args(self, algo):
        args = []
        for coefficient_entry in self.coefficient_entries:
            args.append(coefficient_entry.get_expr())
        if self.is_parametered(algo):
            args.extend(self.get_parameters())

        return args

    def execute_algo(self, algo, *args):
        return algo["algo"](*self.get_args(algo), *args)

    def matrix_eval_expr(self, expr, xx, yy, n):
        func = lambdify((x, y, t), expr, "numpy")

        result = func(xx, yy,
                      np.array([[0.0 for i in range(n)] for j in range(n)]))

        if not expr.is_constant():
            return result
        else:
            return result * np.ones(shape=(n, n))

    def init(self, ht, X, Y, xx, yy, algo):
        self.ax_2D.set_xlim(algo["space"]["x_domain"][0],
                            algo["space"]["x_domain"][1])
        self.ax_2D.set_ylim(algo["space"]["y_domain"][0],
                            algo["space"]["y_domain"][1])

        self.solutions = deque(algo["init"](
            *self.get_args(algo),
            ht,
            0.0,
            X,
            Y,
            *[
                self.matrix_eval_expr(e.get_expr(), xx, yy, len(X))
                for e in self.condition_entries
            ],
        ))

        self.ax_3D.axes.set_zlim3d(bottom=np.min(self.solutions[0]),
                                   top=np.max(self.solutions[0]))

        self.cont = self.ax_2D.contourf(xx,
                                        yy,
                                        self.solutions[0],
                                        cmap=cm.gnuplot)
        self.surface = self.ax_3D.plot_surface(xx,
                                               yy,
                                               self.solutions[0],
                                               cmap=cm.gnuplot)
        self.surface._facecolors2d = self.surface._facecolors3d
        self.surface._edgecolors2d = self.surface._edgecolors3d

        return (self.surface, *self.cont.collections)

    def update(self, ht, t, X, Y, xx, yy, algo):
        self.solutions[-1] = self.execute_algo(algo, ht, t, X, Y,
                                               *self.solutions)
        self.solutions.rotate(1)

        for c in self.cont.collections:
            c.remove()
        self.surface.remove()

        self.cont = self.ax_2D.contourf(xx,
                                        yy,
                                        self.solutions[0],
                                        cmap=cm.gnuplot)
        self.surface = self.ax_3D.plot_surface(xx,
                                               yy,
                                               self.solutions[0],
                                               cmap=cm.gnuplot)
        self.surface._facecolors2d = self.surface._facecolors3d
        self.surface._edgecolors2d = self.surface._edgecolors3d

        return (self.surface, *self.cont.collections)

    def plot(self):
        self.ax_2D.clear()
        self.ax_3D.clear()

        algo = self.compute_algorithms[self.PDE_type.get()]

        x_domain, y_domain = algo["space"]["x_domain"], algo["space"][
            "y_domain"]
        X = np.linspace(x_domain[0], x_domain[1], self.slider_resolution.get())
        Y = np.linspace(y_domain[0], y_domain[1], self.slider_resolution.get())
        xx, yy = np.meshgrid(X, Y)
        ht = algo["time"]["step"]

        if self.is_timed(algo):
            self.ani = animation.FuncAnimation(
                self.fig,
                lambda t: self.update(ht, t, X, Y, xx, yy, algo),
                frames=np.arange(0, algo["time"]["duration"], ht),
                init_func=lambda: self.init(ht, X, Y, xx, yy, algo),
                blit=False,
                repeat=False,
            )
            self.plot_canvas.draw()
        else:
            self.fig.clear()

            solution = self.execute_algo(algo, X, Y)

    def reset(self):
        pass

    def stop(self):
        pass

    def start(self):
        self.plot()

    def is_timed(self, algo):
        return "time" in algo

    def is_parametered(self, algo):
        return "parameters" in algo

    def has_buttons(self, algo):
        return "buttons" in algo

    def add_parameters(self, parameters):
        for parameter, parameter_options in parameters.items():
            if parameter_options["type"] == "slider":
                label_parameter = Label(self.frame_parameters,
                                        text=f"{parameter}: ")
                parameter_widget = Scale(
                    self.frame_parameters,
                    from_=parameter_options["from"],
                    to=parameter_options["to"],
                    resolution=parameter_options["resolution"],
                    orient=HORIZONTAL,
                    bg="#e1e1e1",
                    command=lambda x: self.draw_PDE(),
                )
                parameter_widget.set(parameter_options["default"])
                label_parameter.pack(side=LEFT)
                parameter_widget.pack(fill="x")
                self.label_parameters.append(label_parameter)

            elif parameter_options["type"] == "check":
                parameter_widget = ttk.Checkbutton(
                    self.frame_parameters,
                    text=parameter,
                    onvalue=True,
                    offvalue=False,
                    command=self.draw_PDE,
                )
                parameter_widget.pack(fill="x")

            elif parameter_options["type"] == "expression":
                parameter_widget = LabeledExprEntry(self.frame_parameters)
                parameter_widget.set_text(parameter)
                parameter_widget.pack(fill="x")

            self.parameters.append(
                (parameter, parameter_options, parameter_widget))

    def clear_parameters(self):
        for _, _, parameter in self.parameters:
            parameter.destroy()
        for label in self.label_parameters:
            label.destroy()

        self.label_parameters = []
        self.parameters = []

    def add_buttons(self, buttons):
        for name, button_options in buttons.items():
            button = Button(
                self.frame_buttons,
                text=name,
                command=lambda: button_options["command"]
                (self, **self.get_args(None, self.get_parameters())),
            )
            button.pack(fill="x")

    def clear_buttons(self):
        for buttons in self.frame_buttons.winfo_children():
            buttons.destroy()

    def add_coefficient_entries(self, algo):
        if self.is_timed(algo):
            # Time derivatives coefficients
            for coeff_name in algo["time"]["coefficients"]:
                coefficients_entry = LabeledExprEntry(
                    self.frame_PDE_coefficients)
                coefficients_entry.set_text(coeff_name + "=")
                coefficients_entry.pack(side="left")
                self.coefficient_entries.append(coefficients_entry)

        # Spatial derivatives coefficients
        for coeff_name in algo["space"]["coefficients"]:
            coefficients_entry = LabeledExprEntry(self.frame_PDE_coefficients)
            coefficients_entry.set_text(coeff_name + "=")
            coefficients_entry.pack(side="left")
            self.coefficient_entries.append(coefficients_entry)

        # Function coefficients
        function_coefficients_entry = LabeledExprEntry(
            self.frame_PDE_coefficients)
        function_coefficients_entry.set_text(algo["function"] + "=")
        function_coefficients_entry.pack(side="left")
        self.coefficient_entries.append(function_coefficients_entry)

        # Right side term expression
        right_side_expr_entry = LabeledExprEntry(self.frame_PDE_coefficients)
        right_side_expr_entry.set_text(algo["right side term"] + "=")
        right_side_expr_entry.pack(side="left")
        self.coefficient_entries.append(right_side_expr_entry)

    def clear_coefficients(self):
        for coefficient in self.frame_PDE_coefficients.winfo_children():
            coefficient.destroy()

        self.coefficient_entries = []

    def add_condition_entries(self, algo):
        for i, condition_name in enumerate(algo["conditions"]):
            condition_expr_entry = LabeledExprEntry(self.frame_conditions_expr)
            condition_expr_entry.set_text(condition_name)
            condition_expr_entry.pack(fill="x")
            self.condition_entries.append(condition_expr_entry)

    def clear_conditions(self):
        for condition in self.frame_conditions_expr.winfo_children():
            condition.destroy()

        self.condition_entries = []

    # Event handling
    def handle_algo_selection(self):
        self.clear_conditions()
        self.clear_coefficients()
        self.clear_parameters()
        self.frame_parameters.pack_forget()
        self.frame_buttons.pack_forget()
        self.clear_buttons()
        self.frame_plots.pack_forget()

        i = self.PDE_type.get()
        self.add_coefficient_entries(self.compute_algorithms[i])
        self.add_condition_entries(self.compute_algorithms[i])
        self.set_PDE_equation(self.compute_algorithms[i]["equation"])

        if self.is_parametered(self.compute_algorithms[i]):
            self.add_parameters(self.compute_algorithms[i]["parameters"])
            self.frame_parameters.pack(fill="x")
        if self.has_buttons(self.compute_algorithms[i]):
            self.add_buttons(self.compute_algorithms[i]["buttons"])
            self.frame_buttons.pack(fill="x")
