# mc/io/results.py
import os
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt

def get_project_root():
    """
    Returns the Montecarlo project root directory
    (the folder containing 'mc' and 'main.py')
    """
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..")
    )


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_results_dirs():
    root = get_project_root()
    results = os.path.join(root, "results")
    hist_dir = os.path.join(results, "histograms")
    csv_dir  = os.path.join(results, "csv")

    ensure_dir(hist_dir)
    ensure_dir(csv_dir)

    return hist_dir, csv_dir


def save_histogram(E, weights, phi, bins=80):
    hist_dir, _ = get_results_dirs()

    fname = f"E_spectrum_{np.degrees(phi):.1f}deg.png"
    path = os.path.join(hist_dir, fname)

    plt.figure()
    plt.hist(E, bins=bins, histtype="step", weights=weights)
    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts")
    plt.title(f"Energy deposited ({np.degrees(phi):.1f}°)")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved histogram → {path}")


def save_csv(E_deposited, phi):
    _, csv_dir = get_results_dirs()

    fname = f"E_deposited_{np.degrees(phi):.1f}deg.csv"
    path = os.path.join(csv_dir, fname)

    np.savetxt(
        path,
        E_deposited,
        delimiter=",",
        header="E_deposited_keV",
        comments=""
    )

    print(f"Saved CSV → {path}")
