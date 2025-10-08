import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

# Parameters for the quadratic function
A_DEFAULT = 5.0
STEP_SEQ_DEFAULT = "0.95,4.95,0.95"
N_STEPS = 20

# Function definition
f = lambda x, y, a: x**2 + a * y**2

def grad(x, y, a):
    return np.array([2*x, 2*a*y])

def parse_step_sequence(seq_str):
    try:
        return [float(s) for s in seq_str.split(",") if s.strip()]
    except Exception:
        return [2.9, 1.5]

def gradient_descent_periodic(x0, y0, a, step_seq, n_steps=N_STEPS):
    path = [(x0, y0)]
    x, y = x0, y0
    lipschitz_norm = 2 * a
    for i in range(n_steps):
        g = grad(x, y, a)
        step_len = step_seq[i % len(step_seq)] / lipschitz_norm
        x -= step_len * g[0]
        y -= step_len * g[1]
        path.append((x, y))
    return np.array(path)

# Initial guess
x0, y0 = 2.0, 2.0

def plot_demo_periodic(a, step_seq_str):
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    plt.subplots_adjust(left=0.08, right=0.97, bottom=0.18, wspace=0.32)
    ax2.set_title("Convergence Rate: $||x||$ vs Iteration")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("log(||x||)")

    # Contour plot
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y, a)
    cs = [ax.contour(X, Y, Z, levels=20)]
    ax.set_title(f"Gradient Descent (Periodic Steps) on $f(x, y) = x^2 + {a}y^2$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')

    # Initial guess point
    point, = ax.plot([x0], [y0], 'ro', markersize=8, label="Initial Guess", zorder=3)

    # Step sequence
    step_seq = parse_step_sequence(step_seq_str)

    # Gradient descent path (drawn after contours)
    path = gradient_descent_periodic(x0, y0, a, step_seq)
    line, = ax.plot(path[:,0], path[:,1], 'r--', label="Gradient Descent Path", zorder=4)
    ax.legend()

    # Convergence plot
    errors = np.linalg.norm(path, axis=1)
    conv_line, = ax2.plot(range(len(errors)), np.log(errors), 'b.-', label="Error")
    ax2.legend()

    # Sliders for 'a'
    ax_a = plt.axes([0.1, 0.04, 0.8, 0.03])
    slider_a = Slider(ax_a, 'a', 1.0, 10.0, valinit=a)

    # TextBox for step sequence
    ax_seq = plt.axes([0.1, 0.01, 0.8, 0.03])
    textbox_seq = TextBox(ax_seq, 'Step Seq', initial=step_seq_str)

    # Dragging event
    dragging = {'active': False}
    def on_press(event):
        if event.inaxes != ax: return
        contains, _ = point.contains(event)
        if contains:
            dragging['active'] = True
    def on_motion(event):
        if not dragging['active']: return
        if event.inaxes != ax: return
        x, y = event.xdata, event.ydata
        point.set_data([x], [y])
        step_seq = parse_step_sequence(textbox_seq.text)
        path = gradient_descent_periodic(x, y, slider_a.val, step_seq)
        line.set_data(path[:,0], path[:,1])
        errors = np.linalg.norm(path, axis=1)
        conv_line.set_data(range(len(errors)), np.log(errors))
        ax2.relim()
        ax2.autoscale_view()
        fig.canvas.draw_idle()
    def on_release(event):
        dragging['active'] = False
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    # Slider/TextBox update
    def update(val):
        step_seq = parse_step_sequence(textbox_seq.text)
        a = slider_a.val
        Z = f(X, Y, a)
        # Remove old contours
        for c in cs[0].collections:
            c.remove()
        # Redraw contours and update cs reference
        cs[0] = ax.contour(X, Y, Z, levels=20)
        x, y = point.get_data()
        path = gradient_descent_periodic(x[0], y[0], a, step_seq)
        line.set_data(path[:,0], path[:,1])
        errors = np.linalg.norm(path, axis=1)
        conv_line.set_data(range(len(errors)), np.log(errors))
        ax2.relim()
        ax2.autoscale_view()
        fig.canvas.draw_idle()
    slider_a.on_changed(update)
    textbox_seq.on_submit(update)

    plt.show()

if __name__ == "__main__":
    plot_demo_periodic(A_DEFAULT, STEP_SEQ_DEFAULT)
