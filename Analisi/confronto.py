# Analisi/confronto_spettri.py

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Confronto tra spettro sperimentale calibrato e spettro Monte Carlo."
    )
    parser.add_argument(
        "angle",
        type=str,
        help="Angolo in gradi (es: 30, 60, 90)"
    )
    args = parser.parse_args()
    angle = args.angle


    try:
        # ===============================
        # Percorsi
        # ===============================
        analisi_dir = Path(__file__).parent
        spettro_dati_path = (
            analisi_dir
            / "Analisi_angoli"
            / "Spettri_calibrati"
            / f"{angle}deg_calibrato.csv"
        )

        spettro_simulato_path = (
            analisi_dir.parent
            / "Montecarlo"
            / "results"
            / "csv"
            / f"E_deposited_{angle}deg.csv"
        )

        salva_fig_path = (
            analisi_dir
            / "Analisi_angoli"
            / "Confronto_spettri"
            / f"{angle}deg_confronto.png"
        )

        if not spettro_dati_path.is_file():
            raise FileNotFoundError(f"File non trovato: {spettro_dati_path.resolve()}")

        if not spettro_simulato_path.is_file():
            raise FileNotFoundError(f"File non trovato: {spettro_simulato_path.resolve()}")

        # ===============================
        # Lettura dati
        # ===============================
        spettro_dati = pd.read_csv(spettro_dati_path).iloc[:, 0].to_numpy()
        spettro_simulato = pd.read_csv(spettro_simulato_path).iloc[:, 0].to_numpy()

        print(f"Spettro dati: {spettro_dati.size} eventi")
        print(f"Spettro simulato: {spettro_simulato.size} eventi")

        # ===============================
        # Bins comuni
        # ===============================
        e_min = min(spettro_dati.min(), spettro_simulato.min())
        e_max = max(spettro_dati.max(), spettro_simulato.max())

        bins = np.linspace(e_min, e_max, 80)

        # ===============================
        # Plot
        # ===============================
        plt.figure(figsize=(8, 5))

        plt.hist(
            spettro_dati,
            bins=bins,
            density=True,
            histtype="step",
            label="Dati sperimentali"
        )

        plt.hist(
            spettro_simulato,
            bins=bins,
            density=True,
            histtype="step",
            label="Monte Carlo"
        )

        plt.xlabel("Energia [keV]")
        plt.ylabel("Densità normalizzata")
        plt.title(f"Confronto spettri: {angle}°")
        plt.legend()
        plt.tight_layout()
        plt.savefig(salva_fig_path, dpi=300)
        print(f"Salvato istogramma → {salva_fig_path}")
        plt.show()


    except Exception as e:
        print(f"Errore: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
