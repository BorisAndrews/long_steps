import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, CheckButtons
from matplotlib.widgets import TextBox, Button


# Defaults
DIM_DEFAULT = 256
N_STEPS_DEFAULT = 512
SEQ_DEFAULT = "1.4,2.0,1.4,3.9,1.4,2.0,1.4,7.2,1.4,2.0,1.4,3.9,1.4,2.0,1.4,12.6,1.4,2.0,1.4,3.9,1.4,2.0,1.4,7.2,1.4,2.0,1.4,3.9,1.4,2.0,1.4,23.5,1.4,2.0,1.4,3.9,1.4,2.0,1.4,7.2,1.4,2.0,1.4,3.9,1.4,2.0,1.4,12.6,1.4,2.0,1.4,3.9,1.4,2.0,1.4,7.2,1.4,2.0,1.4,3.9,1.4,2.0,1.4,370.0,1.4,2.0,1.4,3.9,1.4,2.0,1.4,7.2,1.4,2.0,1.4,3.9,1.4,2.0,1.4,12.6,1.4,2.0,1.4,3.9,1.4,2.0,1.4,7.2,1.4,2.0,1.4,3.9,1.4,2.0,1.4,23.5,1.4,2.0,1.4,3.9,1.4,2.0,1.4,7.2,1.4,2.0,1.4,3.9,1.4,2.0,1.4,12.6,1.4,2.0,1.4,3.9,1.4,2.0,1.4,7.2,1.4,2.0,1.4,3.9,1.4,2.0,1.4"
SEED_DEFAULT = None  # set to an int for reproducibility


# f(x) = 1/2 x^T A x, with A PSD (symmetric positive semidefinite)


def make_psd_matrix(dim: int, seed=None):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((dim, dim))
    A = M.T @ M  # PSD
    return A


def make_alt_matrix(dim: int):
    # Inverse discrete Laplacian
    A = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            A[i,j] = (dim + 1) - max(i, j)
    return A


def grad(A: np.ndarray, x: np.ndarray) -> np.ndarray:
    return A @ x


def line_search_step(A: np.ndarray, x: np.ndarray) -> float:
    g = grad(A, x)
    gg = float(g.T @ g)
    gAg = float(g.T @ (A @ g))
    if gAg <= 0 or gg == 0:
        return 0.0
    return gg / gAg


def parse_step_sequence(seq_str: str):
    try:
        seq = [float(s.strip()) for s in seq_str.split(',') if s.strip()]
        if len(seq) == 0:
            return [2.9, 1.5]
        return seq
    except Exception:
        return [2.9, 1.5]


def lambda_max(A: np.ndarray) -> float:
    # Spectral norm equals largest eigenvalue for symmetric PSD A
    return float(np.linalg.norm(A, 2))


def gd_linesearch(A: np.ndarray, x0: np.ndarray, n_steps: int):
    xs = [x0.copy()]
    x = x0.copy()
    for _ in range(n_steps):
        a = line_search_step(A, x)
        if a <= 0:
            xs.append(x.copy())
            break
        x = x - a * grad(A, x)
        xs.append(x.copy())
    return np.array(xs)


def gd_periodic(A: np.ndarray, x0: np.ndarray, step_seq, n_steps: int, scale=True):
    xs = [x0.copy()]
    x = x0.copy()
    lam = lambda_max(A) if scale else 1.0
    seq = list(step_seq)
    L = len(seq)
    if L == 0:
        seq = [2.9, 1.5]
        L = 2
    for k in range(n_steps):
        g = grad(A, x)
        step_len = seq[k % L] / lam
        x = x - step_len * g
        xs.append(x.copy())
    return np.array(xs)


def custom_steps_random(dim, n_steps, r, p):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    s = 2.0 / (np.cosh(np.log(r)) + 1) if r > 0 else 0.0
    steps = []
    n = 0
    rng = np.random.default_rng()
    for k in range(n_steps):
        if rng.uniform() < p:
            steps.append(1.0)
        else:
            steps.append(1.0 / (1.0 - s * np.cos(phi * np.pi * (n+1))**2))
            n += 1
    return steps


def random_theta_steps(dim, n_steps, r, p):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    s = 2.0 / (np.cosh(np.log(r)) + 1) if r > 0 else 0.0
    steps = []
    n = 0
    rng = np.random.default_rng()
    for k in range(n_steps):
        if rng.uniform() < p:
            steps.append(1.0)
        else:
            steps.append(1.0 / (1.0 - s * np.cos(rng.uniform(0, np.pi/2))**2))
            n += 1
    return steps


def errors_norm(xs: np.ndarray) -> np.ndarray:
    return np.linalg.norm(xs, axis=1)


def plot_nd_demo():
    # Initialize state

    dim = DIM_DEFAULT
    n_steps = N_STEPS_DEFAULT
    seq_str = SEQ_DEFAULT
    r_val = 0.5
    p_val = 0.0
    seed = SEED_DEFAULT
    use_alt = [False]  # mutable for closure
    A = make_psd_matrix(dim, seed)
    x0 = np.zeros(dim)
    x0[0] = 1.0

    show_linesearch = [True]
    show_randtheta = [True]
    xs_ls = gd_linesearch(A, x0, n_steps) if show_linesearch[0] else None
    xs_seq = gd_periodic(A, x0, parse_step_sequence(seq_str), n_steps, scale=True)
    xs_custom = gd_periodic(A, x0, custom_steps_random(dim, n_steps, r_val, p_val), n_steps, scale=True)
    xs_randtheta = gd_periodic(A, x0, random_theta_steps(dim, n_steps, r_val, p_val), n_steps, scale=True) if show_randtheta[0] else None
    err_ls = errors_norm(xs_ls) if xs_ls is not None else None
    err_seq = errors_norm(xs_seq)
    err_custom = errors_norm(xs_custom)
    err_randtheta = errors_norm(xs_randtheta) if xs_randtheta is not None else None

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Convergence: ||x|| vs iteration (log y)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("||x||")
    ax.set_yscale('log')

    if show_linesearch[0]:
        line_ls, = ax.plot(range(len(err_ls)), err_ls, 'b.-', label='Line search')
    else:
        line_ls = None
    line_seq, = ax.plot(range(len(err_seq)), err_seq, 'r.-', label='Periodic')
    line_custom, = ax.plot(range(len(err_custom)), err_custom, 'g.-', label='Custom')
    if show_randtheta[0]:
        line_randtheta, = ax.plot(range(len(err_randtheta)), err_randtheta, 'm.-', label='Random θ')
    else:
        line_randtheta = None
    ax_ls = plt.axes([0.52, 0.29, 0.18, 0.06])
    cb_ls = CheckButtons(ax_ls, ['Show Line Search'], [show_linesearch[0]])
    ax_randtheta = plt.axes([0.72, 0.29, 0.18, 0.06])
    cb_randtheta = CheckButtons(ax_randtheta, ['Show Random θ'], [show_randtheta[0]])

    def toggle_linesearch(label):
        nonlocal A, dim, n_steps, seq_str, r_val, p_val, line_ls
        show_linesearch[0] = not show_linesearch[0]
        seq_local = parse_step_sequence(tb_seq.text)
        x0_local = np.zeros(dim)
        x0_local[0] = 1.0
        recompute(A, dim, n_steps, seq_local, r_val, p_val)
    cb_ls.on_clicked(toggle_linesearch)

    def toggle_randtheta(label):
        nonlocal A, dim, n_steps, seq_str, r_val, p_val, line_randtheta
        show_randtheta[0] = not show_randtheta[0]
        seq_local = parse_step_sequence(tb_seq.text)
        x0_local = np.zeros(dim)
        x0_local[0] = 1.0
        recompute(A, dim, n_steps, seq_local, r_val, p_val)
    cb_randtheta.on_clicked(toggle_randtheta)
    # ax.set_ylim(top=2)  # Remove fixed y-axis cap
    ax.legend()

    plt.subplots_adjust(left=0.08, right=0.97, bottom=0.38)
    ax_dim = plt.axes([0.08, 0.22, 0.18, 0.06])
    ax_steps = plt.axes([0.30, 0.22, 0.18, 0.06])
    ax_seq = plt.axes([0.52, 0.22, 0.28, 0.06])
    ax_r = plt.axes([0.08, 0.12, 0.18, 0.06])
    ax_p = plt.axes([0.30, 0.12, 0.18, 0.06])
    ax_alt = plt.axes([0.82, 0.22, 0.14, 0.06])

    tb_dim = TextBox(ax_dim, 'Dim', initial=str(dim))
    tb_steps = TextBox(ax_steps, 'Steps', initial=str(n_steps))
    tb_seq = TextBox(ax_seq, 'Step seq', initial=seq_str)
    tb_r = TextBox(ax_r, 'r [0,1]', initial=str(r_val))
    tb_p = TextBox(ax_p, 'p [0,1]', initial=str(p_val))
    btn_alt = Button(ax_alt, 'Use inverse Laplacian')

    ax_regen = plt.axes([0.52, 0.12, 0.18, 0.06])
    btn_regen = Button(ax_regen, 'Regenerate A')

    def set_alt(event):
        nonlocal A, dim, n_steps, seq_str, r_val, p_val
        use_alt[0] = not use_alt[0]
        if use_alt[0]:
            A = make_alt_matrix(dim)
            btn_alt.label.set_text('Use Random')
        else:
            A = make_psd_matrix(dim, seed)
            btn_alt.label.set_text('Use inverse Laplacian')
        seq_local = parse_step_sequence(tb_seq.text)
        recompute(A, dim, n_steps, seq_local, r_val, p_val)
    btn_alt.on_clicked(set_alt)

    def recompute(A_local, dim_local, n_local, seq_local, r_local, p_local):
        nonlocal line_ls, line_seq, line_custom, line_randtheta, A, dim, n_steps, seq_str, r_val, p_val
        # Update state variables
        A = A_local
        dim = dim_local
        n_steps = n_local
        seq_str = ','.join(str(s) for s in seq_local)
        r_val = r_local
        p_val = p_local
        x0_local = np.zeros(dim_local)
        x0_local[0] = 1.0
        xs_ls_local = gd_linesearch(A_local, x0_local, n_local) if show_linesearch[0] else None
        xs_seq_local = gd_periodic(A_local, x0_local, seq_local, n_local, scale=True)
        xs_custom_local = gd_periodic(A_local, x0_local, custom_steps_random(dim_local, n_local, r_local, p_local), n_local, scale=True)
        xs_randtheta_local = gd_periodic(A_local, x0_local, random_theta_steps(dim_local, n_local, r_local, p_local), n_local, scale=True) if show_randtheta[0] else None
        err_ls_local = errors_norm(xs_ls_local) if xs_ls_local is not None else None
        err_seq_local = errors_norm(xs_seq_local)
        err_custom_local = errors_norm(xs_custom_local)
        err_randtheta_local = errors_norm(xs_randtheta_local) if xs_randtheta_local is not None else None
        # Remove and recreate lines if needed (for legend refresh)
        if show_linesearch[0]:
            if line_ls is None or not hasattr(line_ls, 'set_data'):
                if line_ls is not None:
                    line_ls.remove()
                line_ls, = ax.plot(range(len(err_ls_local)), err_ls_local, 'b.-', label='Line search')
            else:
                line_ls.set_data(range(len(err_ls_local)), err_ls_local)
        else:
            if line_ls is not None:
                line_ls.set_data([], [])
        if line_seq is None or not hasattr(line_seq, 'set_data'):
            if line_seq is not None:
                line_seq.remove()
            line_seq, = ax.plot(range(len(err_seq_local)), err_seq_local, 'r.-', label='Periodic')
        else:
            line_seq.set_data(range(len(err_seq_local)), err_seq_local)
        if line_custom is None or not hasattr(line_custom, 'set_data'):
            if line_custom is not None:
                line_custom.remove()
            line_custom, = ax.plot(range(len(err_custom_local)), err_custom_local, 'g.-', label='Custom')
        else:
            line_custom.set_data(range(len(err_custom_local)), err_custom_local)
        if show_randtheta[0]:
            if err_randtheta_local is not None:
                if line_randtheta is None or not hasattr(line_randtheta, 'set_data'):
                    if line_randtheta is not None:
                        line_randtheta.remove()
                    line_randtheta, = ax.plot(range(len(err_randtheta_local)), err_randtheta_local, 'm.-', label='Random θ')
                else:
                    line_randtheta.set_data(range(len(err_randtheta_local)), err_randtheta_local)
        else:
            if line_randtheta is not None:
                line_randtheta.set_data([], [])
        ax.relim()
        ax.autoscale_view()
        ax.legend()
        fig.canvas.draw_idle()

    def parse_int(s, default):
        try:
            v = int(float(s))
            return max(1, v)
        except Exception:
            return default

    def parse_float(s, default):
        try:
            v = float(s)
            return min(max(v, 0.0), 1.0)
        except Exception:
            return default

    def on_seq_submit(text):
        nonlocal A, dim, n_steps, r_val, p_val
        seq_local = parse_step_sequence(text)
        recompute(A, dim, n_steps, seq_local, r_val, p_val)

    def on_dim_submit(text):
        nonlocal A, dim, n_steps, seq_str, r_val, p_val, use_alt
        dim_new = parse_int(text, dim)
        dim = dim_new
        if use_alt[0]:
            A = make_alt_matrix(dim)
        else:
            A = make_psd_matrix(dim, seed)
        seq_local = parse_step_sequence(tb_seq.text)
        recompute(A, dim, n_steps, seq_local, r_val, p_val)

    def on_steps_submit(text):
        nonlocal n_steps, A, dim, seq_str, r_val, p_val, use_alt
        n_steps_new = parse_int(text, n_steps)
        n_steps = n_steps_new
        seq_local = parse_step_sequence(tb_seq.text)
        recompute(A, dim, n_steps, seq_local, r_val, p_val)

    def on_r_submit(text):
        nonlocal r_val, A, dim, n_steps, seq_str, p_val
        r_val_new = parse_float(text, r_val)
        r_val = r_val_new
        seq_local = parse_step_sequence(tb_seq.text)
        recompute(A, dim, n_steps, seq_local, r_val, p_val)

    def on_p_submit(text):
        nonlocal p_val, A, dim, n_steps, seq_str, r_val
        p_val_new = parse_float(text, p_val)
        p_val = p_val_new
        seq_local = parse_step_sequence(tb_seq.text)
        recompute(A, dim, n_steps, seq_local, r_val, p_val)

    def on_regen(event):
        nonlocal A, dim, n_steps, seq_str, r_val, p_val, use_alt
        if use_alt[0]:
            A = make_alt_matrix(dim)
        else:
            A = make_psd_matrix(dim, seed)
        seq_local = parse_step_sequence(tb_seq.text)
        recompute(A, dim, n_steps, seq_local, r_val, p_val)

    tb_seq.on_submit(on_seq_submit)
    tb_dim.on_submit(on_dim_submit)
    tb_steps.on_submit(on_steps_submit)
    tb_r.on_submit(on_r_submit)
    tb_p.on_submit(on_p_submit)
    btn_regen.on_clicked(on_regen)

    plt.show()

if __name__ == '__main__':
    plot_nd_demo()
