import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Parameters for the quadratic function
A_DEFAULT = 5.0
N_STEPS = 20

# Function definition
f = lambda x, y, a: x**2 + a * y**2

def grad(x, y, a):
    return np.array([2*x, 2*a*y])

def linesearch_step(x, y, a):
    g = grad(x, y, a)
    # Analytical line search for quadratic: optimal step size = (g^T g) / (g^T H g), H = diag([2, 2a])
    H = np.array([[2, 0], [0, 2*a]])
    gTg = np.dot(g, g)
    gTHg = np.dot(g, H @ g)
    if gTHg == 0:
        return 0.1  # fallback
    return gTg / gTHg

def gradient_descent_linesearch(x0, y0, a, n_steps=N_STEPS):
    path = [(x0, y0)]
    x, y = x0, y0
    step_sizes = []
    for i in range(n_steps):
        g = grad(x, y, a)
        alpha = linesearch_step(x, y, a)
        # Normalized step size
        lipschitz_norm = 2 * a
        normalized_step = alpha * lipschitz_norm
        step_sizes.append(normalized_step)
        x -= alpha * g[0]
        y -= alpha * g[1]
        path.append((x, y))
    return np.array(path), step_sizes

# Initial guess
x0, y0 = 2.0, 2.0

def plot_demo_linesearch(a):
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(12, 5))
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
    ax.set_title(f"Gradient Descent with Line Search on $f(x, y) = x^2 + {a}y^2$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')

    # Initial guess point
    point, = ax.plot([x0], [y0], 'ro', markersize=8, label="Initial Guess", zorder=3)

    # Gradient descent path (drawn after contours)
    path, step_sizes = gradient_descent_linesearch(x0, y0, a)
    line, = ax.plot(path[:,0], path[:,1], 'r--', label="Line Search Path", zorder=4)
    ax.legend()

    # Convergence plot
    errors = np.linalg.norm(path, axis=1)
    conv_line, = ax2.plot(range(len(errors)), np.log(errors), 'b.-', label="Error")
    ax2.legend()

    # Text box for step sizes (move to right of convergence plot)
    ax_text = fig.add_axes([0.78, 0.18, 0.19, 0.75])
    ax_text.axis('off')
    show_steps = [True]
    text_obj = ax_text.text(0, 1, '', va='top', ha='left', fontsize=10, family='monospace', visible=True)

    def update_text():
        if show_steps[0]:
            lines = ["Step | Normalized Step Size"]
            for i, s in enumerate(step_sizes):
                lines.append(f"{i+1:4d} | {s:.1f}")
            text_obj.set_text('\n'.join(lines))
            text_obj.set_visible(True)
        else:
            text_obj.set_visible(False)
        fig.canvas.draw_idle()
    update_text()

    # Toggle button (move below table)
    from matplotlib.widgets import Button
    ax_toggle = fig.add_axes([0.78, 0.08, 0.19, 0.07])
    btn_toggle = Button(ax_toggle, 'Toggle Step Sizes')
    def on_toggle(event):
        show_steps[0] = not show_steps[0]
        update_text()
    btn_toggle.on_clicked(on_toggle)

    # Slider for 'a'
    ax_a = plt.axes([0.1, 0.01, 0.8, 0.03])  # Move slider lower to avoid overlap
    slider_a = Slider(ax_a, 'a', 1.0, 10.0, valinit=a)

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
        path_new, step_sizes_new = gradient_descent_linesearch(x, y, slider_a.val)
        line.set_data(path_new[:,0], path_new[:,1])
        errors = np.linalg.norm(path_new, axis=1)
        conv_line.set_data(range(len(errors)), np.log(errors))
        ax2.relim()
        ax2.autoscale_view()
        nonlocal step_sizes
        step_sizes = step_sizes_new
        update_text()
    fig.canvas.draw_idle()
    def on_release(event):
        dragging['active'] = False
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    # Slider update
    def update(val):
        a = slider_a.val
        Z = f(X, Y, a)
        # Remove old contours
        for c in cs[0].collections:
            c.remove()
        # Redraw contours and update cs reference
        cs[0] = ax.contour(X, Y, Z, levels=20)
        x, y = point.get_data()
        path_new, step_sizes_new = gradient_descent_linesearch(x[0], y[0], a)
        line.set_data(path_new[:,0], path_new[:,1])
        errors = np.linalg.norm(path_new, axis=1)
        conv_line.set_data(range(len(errors)), np.log(errors))
        ax2.relim()
        ax2.autoscale_view()
        nonlocal step_sizes
        step_sizes = step_sizes_new
        update_text()
    fig.canvas.draw_idle()
    slider_a.on_changed(update)

    plt.show()

if __name__ == "__main__":
    plot_demo_linesearch(A_DEFAULT)
