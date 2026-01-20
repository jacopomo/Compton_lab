# mc/utils/plotting.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mc.utils.math3d import unpack_stacked

def plot_photon_positions(positions, N_max=None):
    """Plots and shows 3-D scatterplot of photon positions, rotating the axes
    to reflect the laboratory setting and setting a clean aspect ratio.

    Args:
        positions (nparray): (N,3) numpy array of x,y,z photon positions
        N_max (int, optional): maximum number of points to plot.
                              If None, all points are plotted.
    """
    positions = np.asarray(positions)

    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N,3)")

    N = positions.shape[0]

    # --- subsample if needed ---
    if (N_max is not None) and (N > N_max):
        idx = np.random.choice(N, size=int(N_max), replace=False)
        positions = positions[idx]

    x, y, z = unpack_stacked(positions)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x, y, z, s=5, alpha=0.6)

    # Orient axes to laboratory frame
    ax.view_init(elev=285, azim=0)

    # --- equal aspect ratio ---
    max_range = np.array([
        x.max() - x.min(),
        y.max() - y.min(),
        z.max() - z.min()
    ]).max() / 2.0

    mid_x = 0.5 * (x.max() + x.min())
    mid_y = 0.5 * (y.max() + y.min())
    mid_z = 0.5 * (z.max() + z.min())

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()


def plot_entry_exit(
    initial_pos,
    final_pos,
    Rmax=None,
    Nx=120,
    Ny=120,
    r_bins=80,
    n_circles=6
):
    """
    Create a 2x2 figure with rectangular (x,y) bins (equal-area bins):
      - Left column : Entry
      - Right column: Exit
      - Top row     : 2D histogram in (x,y) with circular grid overlay
      - Bottom row  : 1D histogram of radial coordinate r = sqrt(x^2 + y^2)

    Notes
    -----
    - Using rectangular bins fixes the "different bin area" issue from (r,theta) binning.
    - Zero-count bins are displayed in white by masking them (set to NaN).
    - The circular overlay is purely a visual guide (not the binning).
    """

    # --- extract x,y coordinates ---
    ini = np.asarray(initial_pos)
    fin = np.asarray(final_pos)
    xi, yi = ini[:, 0], ini[:, 1]
    xf, yf = fin[:, 0], fin[:, 1]

    # --- compute radial coordinates ---
    ri = np.sqrt(xi**2 + yi**2)
    rf = np.sqrt(xf**2 + yf**2)

    # --- choose a common maximum radius ---
    if Rmax is None:
        Rmax = float(np.nanmax(np.concatenate([ri, rf])))
        if not np.isfinite(Rmax) or Rmax == 0.0:
            Rmax = 1.0

    # --- define rectangular bin edges in x and y ---
    x_edges = np.linspace(-Rmax, Rmax, Nx + 1)
    y_edges = np.linspace(-Rmax, Rmax, Ny + 1)

    def hist2d_xy(x, y):
        """
        Compute a 2D histogram in (x,y) with equal-area rectangular bins.
        Returns an array H of shape (Ny, Nx) for pcolormesh convenience.
        """
        H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        # histogram2d returns shape (Nx, Ny); transpose to (Ny, Nx)
        return H.T

    # --- 2D histograms ---
    Hi = hist2d_xy(xi, yi)
    Hf = hist2d_xy(xf, yf)

    # --- mask zero-count bins to show them as white ---
    Hi_plot = Hi.astype(float).copy()
    Hf_plot = Hf.astype(float).copy()
    Hi_plot[Hi_plot == 0] = np.nan
    Hf_plot[Hf_plot == 0] = np.nan

    # --- build mesh for pcolormesh ---
    Xg, Yg = np.meshgrid(x_edges, y_edges, indexing="xy")

    # --- create figure and axes ---
    fig, axs = plt.subplots(
        2, 2, figsize=(12, 10),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True
    )

    # --- column titles ---
    axs[0, 0].set_title("Entrata")
    axs[0, 1].set_title("Uscita")

    # --- colormap setup: NaNs (masked) will be white ---
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    # --- shared color scaling for fair comparison ---
    vmax = np.nanmax([np.nanmax(Hi_plot), np.nanmax(Hf_plot)])
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    # --- top row: 2D plots ---
    m0 = axs[0, 0].pcolormesh(Xg, Yg, Hi_plot, shading="auto", cmap=cmap, vmin=1e-12, vmax=vmax)
    m1 = axs[0, 1].pcolormesh(Xg, Yg, Hf_plot, shading="auto", cmap=cmap, vmin=1e-12, vmax=vmax)

    # --- overlay circular grid + axes formatting ---
    circles_r = np.linspace(0, Rmax, n_circles)[1:]  # skip 0
    for ax in (axs[0, 0], axs[0, 1]):
        ax.set_aspect("equal", "box")
        ax.set_xlim(-Rmax, Rmax)
        ax.set_ylim(-Rmax, Rmax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # circular reference grid (overlay only)
        for rr in circles_r:
            ax.add_patch(plt.Circle((0, 0), rr, fill=False, linewidth=0.8))

        # Cartesian axes
        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linewidth=0.8)

    # --- one shared colorbar for the top row ---
    cbar = fig.colorbar(m1, ax=axs[0, :], shrink=0.9)
    cbar.set_label("Counts per (Δx·Δy) bin")

    # --- bottom row: 1D radial histograms ---
    r_edges = np.linspace(0.0, Rmax, r_bins + 1)
    axs[1, 0].hist(ri, bins=r_edges)
    axs[1, 1].hist(rf, bins=r_edges)

    axs[1, 0].set_xlabel("r = sqrt(x² + y²)")
    axs[1, 1].set_xlabel("r = sqrt(x² + y²)")
    axs[1, 0].set_ylabel("Counts")
    axs[1, 1].set_ylabel("Counts")
    axs[1, 0].set_title("Radiale (Entrata)")
    axs[1, 1].set_title("Radiale (Uscita)")

    plt.show()
    return fig, axs

def hist_radial_vs_energy(
    points,
    energy,
    weights=None,
    center=None,
    normal=None,
    r_bins=80,
    e_bins=80,
    Rmax=None,
    Emin=None,
    Emax=None,
):
    """
    Figure layout:
      - Center: 2D histogram (r vs energy), weighted (sum of weights)
      - Left  : projection on energy axis (weighted), horizontal bars (vertical axis = energy)
      - Bottom: projection on r axis with TWO curves/bars:
               (1) weighted sum of weights vs r
               (2) unweighted counts vs r

    Notes
    -----
    - Points are assumed to lie on a planar circular surface defined by (center, normal).
    - 2D histogram uses weights if provided; if not, weights=1.
    - Zero bins in the 2D map are shown as white.
    """

    points = np.asarray(points, dtype=float)
    energy = np.asarray(energy, dtype=float).reshape(-1)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N,3)")
    if energy.shape[0] != points.shape[0]:
        raise ValueError("energy must have shape (N,) matching points")

    # --- weights ---
    if weights is None:
        w = np.ones(points.shape[0], dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != points.shape[0]:
            raise ValueError("weights must have shape (N,) or (N,1) matching points")

    # --- geometry ---
    center = np.asarray(center, dtype=float).reshape(3)
    n = np.asarray(normal, dtype=float).reshape(3)
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        raise ValueError("normal vector must be non-zero")
    n = n / n_norm

    # --- compute in-plane radial coordinate ---
    v = points - center[None, :]
    proj = np.dot(v, n)                      # (N,)
    v_plane = v - proj[:, None] * n[None, :] # (N,3)
    r = np.linalg.norm(v_plane, axis=1)      # (N,)

    # --- ranges ---
    if Rmax is None:
        Rmax = float(np.nanmax(r)) if np.any(np.isfinite(r)) else 1.0
        if Rmax == 0:
            Rmax = 1.0

    if Emin is None:
        Emin = float(np.nanmin(energy))
    if Emax is None:
        Emax = float(np.nanmax(energy))
    if Emin == Emax:
        Emax = Emin + 1.0

    # --- edges ---
    r_edges = np.linspace(0.0, Rmax, int(r_bins) + 1) if np.isscalar(r_bins) else np.asarray(r_bins, dtype=float)
    e_edges = np.linspace(Emin, Emax, int(e_bins) + 1) if np.isscalar(e_bins) else np.asarray(e_bins, dtype=float)

    # --- 2D histogram: r vs E (weighted) ---
    H2, _, _ = np.histogram2d(r, energy, bins=[r_edges, e_edges], weights=w)  # (Nr, Ne)

    # --- 1D projections from the data (not from H2, but equivalent and simpler) ---
    Hr_w, _ = np.histogram(r, bins=r_edges, weights=w)       # weighted vs r
    Hr_c, _ = np.histogram(r, bins=r_edges, weights=None)    # counts vs r
    He_w, _ = np.histogram(energy, bins=e_edges, weights=w)  # weighted vs E

    # --- plot setup: white for zero bins in 2D map ---
    H2_plot = H2.astype(float).copy()
    H2_plot[H2_plot == 0] = np.nan
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    # --- figure layout (manual axes for clean control) ---
    fig = plt.figure(figsize=(11, 8), constrained_layout=False)

    # Axes rectangles: [left, bottom, width, height] in figure fraction
    ax2d  = fig.add_axes([0.22, 0.20, 0.70, 0.72])  # main heatmap
    axE   = fig.add_axes([0.06, 0.20, 0.14, 0.72], sharey=ax2d)  # left projection
    axR   = fig.add_axes([0.22, 0.06, 0.70, 0.12], sharex=ax2d)  # bottom projection

    # --- main 2D map ---
    m = ax2d.pcolormesh(r_edges, e_edges, H2_plot.T, shading="auto", cmap=cmap)
    ax2d.set_xlabel("Radial position r on surface [cm]")
    ax2d.set_ylabel("Energy")
    ax2d.set_title("2D histogram: r vs energy (weighted)")

    # Colorbar attached to the main axes
    cax = fig.add_axes([0.93, 0.20, 0.02, 0.72])
    cbar = fig.colorbar(m, cax=cax)
    cbar.set_label("Sum of weights")

    # --- left projection: weighted vs energy (horizontal bars) ---
    e_centers = 0.5 * (e_edges[:-1] + e_edges[1:])
    e_heights = (e_edges[1:] - e_edges[:-1])

    axE.barh(e_centers, He_w, height=e_heights, align="center")
    axE.set_xlabel("Σw")
    axE.set_ylabel("")  # shared with ax2d
    axE.tick_params(axis="y", labelleft=False)  # hide duplicated y tick labels
    axE.set_title("Proj(E)")

    # --- bottom projection: r weighted and r unweighted ---
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    r_widths  = (r_edges[1:] - r_edges[:-1])

    # Weighted as bars
    axR.bar(r_centers, Hr_w, width=r_widths, align="center", alpha=0.8, label="Σw vs r")

    # Unweighted counts as a line (clean overlay)
    axR.plot(r_centers, Hr_c, linewidth=1.5, label="Counts vs r")

    axR.set_ylabel("Proj(r)")
    axR.set_xlabel("r [cm]")
    axR.legend(loc="upper right")

    # --- cosmetic: avoid duplicated x tick labels on the main axis ---
    ax2d.tick_params(axis="x", labelbottom=False)

    plt.show()

    return {
        "H2": H2,
        "r_edges": r_edges,
        "e_edges": e_edges,
        "Hr_weighted": Hr_w,
        "Hr_counts": Hr_c,
        "He_weighted": He_w,
        "fig": fig,
        "axes": {"2d": ax2d, "E": axE, "R": axR},
    }

def plot_histogram(data, weights=None, bins=50, range=None, density=False, title=None):
    """
    Plot a 1D histogram from an array of shape (N,1) or (N,),
    optionally using per-event weights.

    Parameters
    ----------
    data : ndarray, shape (N,1) or (N,)
        Input data.
    weights : ndarray, shape (N,1) or (N,), optional
        Per-event weights.
    bins : int or sequence
        Number of bins or explicit bin edges.
    range : tuple, optional
        (min, max) range for the histogram.
    density : bool, optional
        If True, normalize to a probability density.
    title : str, optional
        Title of the plot.
    """

    x = np.asarray(data, dtype=float).reshape(-1)

    if weights is None:
        w = None
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != x.shape[0]:
            raise ValueError("weights must have the same length as data")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        x,
        bins=bins,
        range=range,
        weights=w,
        density=density
    )

    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if density else "Weighted counts")

    if title is not None:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig, ax

def compare_weighted_unweighted_histogram(
    data,
    weights,
    bins=50,
    range=None,
    density=False,
    title=None
):
    """
    Compare weighted and unweighted histograms of the same data
    in a single figure, using the same binning.

    Parameters
    ----------
    data : ndarray, shape (N,) or (N,1)
        Input data.
    weights : ndarray, shape (N,) or (N,1)
        Per-event weights.
    bins : int or sequence
        Number of bins or explicit bin edges.
    range : tuple, optional
        (min, max) histogram range.
    density : bool, optional
        If True, normalize histograms to probability densities.
    title : str, optional
        Figure title.
    """

    x = np.asarray(data, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)

    if x.shape[0] != w.shape[0]:
        raise ValueError("data and weights must have the same length")

    # --- common binning ---
    counts_unw, edges = np.histogram(
        x, bins=bins, range=range, density=density
    )

    counts_w, _ = np.histogram(
        x, bins=edges, weights=w, density=density
    )

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        centers,
        counts_unw,
        width=widths,
        align="center",
        alpha=0.5,
        label="Unweighted",
        edgecolor="black"
    )

    ax.bar(
        centers,
        counts_w,
        width=widths,
        align="center",
        alpha=0.5,
        label="Weighted",
        edgecolor="black"
    )

    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if density else "Counts / Sum of weights")

    if title is not None:
        ax.set_title(title)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {
        "bin_edges": edges,
        "counts_unweighted": counts_unw,
        "counts_weighted": counts_w,
        "figure": fig,
        "axis": ax,
    }

def plot_histogram_Nm_overlay(
    data,
    weights=None,
    bins=50,
    hist_range=None,
    density=False,
    title=None,
    labels=None,
    alpha=0.45,
    histtype="stepfilled",
):
    """
    Overlay m histograms (one per column) on the SAME axes.
    """

    X = np.asarray(data, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("data must have shape (N,m) or (N,)")

    N, m = X.shape

    # --- weights ---
    if weights is None:
        w = None
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != N:
            raise ValueError("weights must have shape (N,) matching data rows")

    # --- labels ---
    if labels is None:
        labels = [f"col {j}" for j in range(m)]
    else:
        if len(labels) != m:
            raise ValueError("labels must have length m")

    fig, ax = plt.subplots(figsize=(8, 5))

    # --- shared bin edges ---
    if np.isscalar(bins):
        if hist_range is None:
            xmin = float(np.nanmin(X))
            xmax = float(np.nanmax(X))
            if xmin == xmax:
                xmax = xmin + 1.0
            edges = np.linspace(xmin, xmax, int(bins) + 1)
        else:
            edges = np.linspace(hist_range[0], hist_range[1], int(bins) + 1)
    else:
        edges = np.asarray(bins, dtype=float)

    # --- overlay histograms ---
    for j in range(m):
        ax.hist(
            X[:, j],
            bins=edges,
            weights=w,
            density=density,
            alpha=alpha if "filled" in histtype or histtype == "bar" else 1.0,
            histtype=histtype,
            label=labels[j],
        )

    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if density else "Counts" if w is None else "Weighted counts")
    if title is not None:
        ax.set_title(title)

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return fig, ax

def plot_3d_linked_by_index(
    *arrays,
    weights=None,
    N_max=None,
    color="tab:blue",
    alpha=0.4,
    linewidth=1.0,
    markersize=10,
    seed=None,
):
    """
    Plot 3D points from multiple (N,3) arrays and connect points
    with the same index across arrays.

    Optional weighted subsampling:
    - points with larger weight are less likely to be discarded.

    Parameters
    ----------
    *arrays : ndarray(s), shape (N,3)
        Arrays of 3D points. All must have the same N.
    weights : ndarray, shape (N,) or (N,1), optional
        Weights used for biased subsampling.
    N_max : int, optional
        Maximum number of trajectories to plot.
    color : color
        Color for lines and points.
    alpha : float
        Transparency for lines.
    linewidth : float
        Line width.
    markersize : float
        Marker size.
    seed : int, optional
        Random seed for reproducibility.
    """

    if len(arrays) < 2:
        raise ValueError("At least two (N,3) arrays are required")

    arrays = [np.asarray(a, dtype=float) for a in arrays]

    N = arrays[0].shape[0]
    for a in arrays:
        if a.ndim != 2 or a.shape != (N, 3):
            raise ValueError("All arrays must have shape (N,3) with the same N")

    # --- handle weights ---
    if weights is not None:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != N:
            raise ValueError("weights must have shape (N,)")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")

        w_sum = np.sum(w)
        if w_sum == 0:
            raise ValueError("sum of weights must be > 0")

        probs = w / w_sum
    else:
        probs = None

    if seed is not None:
        np.random.seed(seed)

    # --- subsample indices ---
    if (N_max is not None) and (N > N_max):
        idx = np.random.choice(
            N,
            size=N_max,
            replace=False,
            p=probs
        )
    else:
        idx = np.arange(N)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # --- plot trajectories ---
    for i in idx:
        pts = np.array([a[i] for a in arrays])  # (M,3)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        ax.plot(x, y, z, color=color, alpha=alpha, linewidth=linewidth)
        ax.scatter(x, y, z, color=color, s=markersize)

    # --- equal aspect ratio ---
    all_pts = np.vstack(arrays)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0
    mid = 0.5 * (all_pts.max(axis=0) + all_pts.min(axis=0))

    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.tight_layout()
    plt.show()

    return fig, ax


def scatter_xy(
    x,
    y,
    title="",
    N_max=None,
    lines=None,
    n_line_points=500,
    alpha=0.6,
    s=10,
):
    """
    Scatter plot of two 1D arrays with optional subsampling and
    optional overlay of analytic functions y = f(x).

    Parameters
    ----------
    x : array-like, shape (N,)
        x values.
    y : array-like, shape (N,)
        y values.
    title : str
        Title of the plot.
    N_max : int, optional
        Maximum number of points to plot (random subsampling).
    lines : list of callables, optional
        List of functions f(x) to be plotted over the x range.
    n_line_points : int
        Number of points used to draw each function.
    alpha : float
        Transparency of scatter points.
    s : float
        Marker size.
    """

    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")

    N = len(x)

    # --- subsample scatter points if needed ---
    if (N_max is not None) and (N > N_max):
        idx = np.random.choice(N, size=N_max, replace=False)
        x_sc = x[idx]
        y_sc = y[idx]
    else:
        x_sc = x
        y_sc = y

    plt.figure(figsize=(6, 5))
    plt.scatter(x_sc, y_sc, alpha=alpha, s=s, label="data")

    # --- plot analytic lines if provided ---
    if lines is not None:
        x_min, x_max = np.min(x), np.max(x)
        x_line = np.linspace(x_min, x_max, n_line_points)

        for f in lines:
            y_line = f(x_line)
            plt.plot(x_line, y_line, linewidth=2)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def hist2d(
    x,
    y,
    bins=50,
    range=None,
    weights=None,
    title=None,
    xlabel="x",
    ylabel="y",
    cmap="viridis",
    density=False,
    show=True
):
    """
    Genera un istogramma 2D a partire da due array 1D.

    Parametri
    ----------
    x, y : array-like
        Dati 1D (stessa lunghezza)
    bins : int o [int, int], opzionale
        Numero di bin (default: 50)
    range : [[xmin, xmax], [ymin, ymax]], opzionale
        Range dell'istogramma
    weights : array-like, opzionale
        Pesi evento-per-evento
    title : str, opzionale
        Titolo della figura
    xlabel, ylabel : str
        Etichette degli assi
    cmap : str
        Colormap
    density : bool
        Normalizzazione a densità
    show : bool
        Se True mostra la figura

    Ritorna
    -------
    H : ndarray
        Contenuto dei bin
    xedges, yedges : ndarray
        Bordi dei bin
    fig, ax : matplotlib objects
    """

    x = np.asarray(x)
    y = np.asarray(y)


    if x.shape != y.shape:
        raise ValueError("x e y devono avere la stessa dimensione")
    
    cmap = plt.get_cmap(cmap)
    newcolors = cmap(np.linspace(0, 1, 256))
    newcolors[0, :] = np.array([1, 1, 1, 0])
    newcmap = ListedColormap(newcolors)

    fig, ax = plt.subplots()

    H, xedges, yedges, img = ax.hist2d(
        x,
        y,
        bins=bins,
        range=range,
        weights=weights,
        density=density,
        cmap=newcmap
    )

    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Counts" if not density else "Density")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if show:
        plt.show()

    return H, xedges, yedges, fig, ax

