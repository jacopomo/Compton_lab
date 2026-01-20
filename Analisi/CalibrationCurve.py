import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.stats import linregress
import json
import os
import inspect
import argparse

import utils as u

# --- 1. MODELLO MATEMATICO ---
functions_list = {"gauss_exp":{"func":u.gauss_exp, "num peak": 1},
                  "double_gauss_exp":{"func":u.double_gauss_exp, "num peak": 2}}


# --- 2. FUNZIONE DI FIT ---
def esegui_fit(bin_centers, counts, config_sorgente, bar_printing=False, visualizzare=False):
    try:
        nome = config_sorgente['nome']
        energia = config_sorgente['energia']
        finestra = config_sorgente['finestra'] # [low, high]
        guess = config_sorgente['guess']
        manual_bounds = config_sorgente.get('bounds')
        special_fit = config_sorgente.get('special fit') # Opzionale
    except Exception as e:
        print(f"\n --> ERRORE: {e}\n            Controlla di aver definito nome, energia, finestra e guess nel file JSON.\n")    

    if not bar_printing:
        print(f"\nAnalisi: {nome} ({energia} keV)...")

    # Selezione Finestra
    low, high = finestra
    mask = (bin_centers >= low) & (bin_centers <= high)
    x_win = bin_centers[mask]
    y_win = counts[mask]

    # Controllo preliminare dati
    if len(x_win) < 6:
        print(f"\n  --> ERRORE: Finestra [{low}-{high}] vuota o troppo stretta!\n")
        return None, None, None, None

    # Scelgo La Funzione Di Fit
    if special_fit is not None:
        try:
            fit_function = functions_list[special_fit]["func"]
            num_peak = functions_list[special_fit]["num peak"]
            pippo = np.arange(num_peak, dtype=int)
            musk = np.ones(len(pippo), dtype=int) + 3 * pippo
            if not bar_printing:
                print(f"  --> Selezionato manualmente la funzione di fit '{special_fit}'")
        except:
            print(f"\n  --> ERRORE:'{special_fit}' non rietra tra le funzioni di fit possibili.\n")
            return None, None, None, None
    else:
        fit_function = functions_list["gauss_exp"]["func"]
        musk = [1]
    muskII = musk + np.ones(len(musk), dtype=int)

    
    #Fisso I Parametri Iniziali
    num_par = len(inspect.signature(fit_function).parameters) - 1
    if len(guess) == num_par:
        p0 = guess
    else:
        print(f"\n  --> ERRORE: Il numero dei parametri iniziali e' sbagliato!\n             La funzione '{fit_function}' richiede {num_par} parametri\n")
        return None, None, None, None
    if manual_bounds is not None:
        try:
            bounds_min = manual_bounds[0]
            bounds_max = manual_bounds[1]
        except:
            print("\n  --> ERRORE: la varibile bounds nel JSON deve essere una ntuple di due elementi.")
            return None, None, None, None
        # Controllo che i limiti e i parametri iniziali abbiamo la stessa dimensione
        if len(bounds_min)!=num_par or len(bounds_max)!=num_par:
            print(f"\n  --> ERRORE: Il numero dei bounds e' sbagliato!\n             La funzione '{fit_function}' richiede {num_par} parametri\n")
            return None, None, None, None
    else:
        if fit_function == u.gauss_exp:
            #             A,      mu,   sigma,  B0,     k,      B1
            bounds_min = [0,      low,  0,      0,      0,      0]
            bounds_max = [np.inf, high, np.inf, np.inf, np.inf, np.inf]
        if fit_function == u.double_gauss_exp:
            #             A1,     mu1,  sigma1, A2,     mu2,  sigma2, B0,     k,      B1
            bounds_min = [0,      low,  0,      0,      low,  0,      0,      0,      0]
            bounds_max = [np.inf, high, np.inf, np.inf, high, np.inf, np.inf, np.inf, np.inf]
    

    # Esecuzione Fit
    try:
        
        popt, pcov = curve_fit(fit_function, x_win, y_win, p0=p0, bounds=(bounds_min, bounds_max), maxfev=20000)
        err_mu = np.sqrt(np.diag(pcov))[musk]
        err_sigma = np.sqrt(np.diag(pcov))[muskII]

        
        # Visualizzazione risultato fit
        fig, ax_top = plt.subplots(figsize=(6, 4))

        # spettro sul pannello principale
        ax_top.step(bin_centers, counts, where='mid', color='lightgray', label='Spettro intero')
        ax_top.plot(x_win, y_win, 'b.', label='Dati Finestra')
        x_plot = np.linspace(low, high, 1000)
        ax_top.plot(x_plot, fit_function(x_plot, *popt), 'r-', linewidth=2, label=f'Fit (mu={popt[1]:.2f})')
        ax_top.set_ylabel("Conteggi")
        ax_top.set_title(f"Fit: {nome}")
        ax_top.legend()

        # crea pannello residui attaccato sotto, senza spazio
        divider   = make_axes_locatable(ax_top)
        ax_bottom = divider.append_axes("bottom", size="25%", pad=0.0, sharex=ax_top)

        # residui
        res = y_win - fit_function(x_win, *popt)
        ax_bottom.plot(x_win, res, 'b.')
        ax_bottom.axhline(0, linestyle='--')
        ax_bottom.set_xlabel("Canale")
        ax_bottom.set_ylabel("Residui")
        ax_bottom.set_xlim(low*0.8, high*1.2)

        # niente label x nel pannello sopra
        ax_top.tick_params(axis='x', labelbottom=False)

        # togli il bordo superiore del pannello residui
        ax_bottom.tick_params(top=False)

        if not bar_printing:
            print(f"  --> OK! Picco trovato a canale {popt[musk]} +/- {err_mu}")
            print(f"  --> Contronto tra parametri di fit e parametri iniziali:")
            print(f"      popt - p0 = {popt - p0}")
            print(f"      popt = {popt}")
        
        if visualizzare:
            plt.show()
        else:
            plt.close()

        return popt[musk], popt[muskII], err_mu, err_sigma

    except Exception as e:
        print(f"  --> FIT FALLITO: {e}")
        return None, None, None, None


def calibration(day, bar_printing=False, fits=False, bar_visualizzare=False):
    # --- 3. LETTURA DATI E CONFIGURAZIONE ---

    base_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "Dati", "Calibration")
    data_path = os.path.join(base_path, day)

    config_file = "config_" + day + ".json"
    config_path = os.path.join(base_path, "Config", config_file)


    if not os.path.exists(config_path):
        print(f"ERRORE: Devi creare il file '{config_file}'!")
        exit()

    with open(config_path, 'r') as f:
        config_globale = json.load(f)

    lista_sorgenti = config_globale['sorgenti']

    enable_print = bool(config_globale["print"])
    enable_show = bool(config_globale["show"])

    if not bar_visualizzare:
        print(f"--- AVVIO CALIBRAZIONE AUTOMATICA ---")

    # --- 4. CICLO DI ANALISI ---
    punti_ch = np.array([])
    punti_E = np.array([])
    errori_ch = np.array([])

    punti_ch_sigma = np.array([])

    nomi_souce = np.array([])
    fit_or_not = np.array([])
    cache_dati = {} # Per non ricaricare i file se usati più volte

    for sorgente in lista_sorgenti:
        nome_file = sorgente['file']
        
        # Caricamento (solo se non già in memoria)
        if nome_file not in cache_dati:
            path = os.path.join(data_path, nome_file)
            if not os.path.exists(path):
                print("\n=====================================================================================")
                print(f"ATTENZIONE: File '{path}' non trovato. Salto {sorgente['nome']}.")
                print("=====================================================================================")
                continue
            
            try:
                resize = sorgente['resize_factor']
                num_bins = int(8192 / resize)

                dat = np.loadtxt(path, dtype=int, unpack=True)
                unbinned = np.repeat(np.arange(dat.size, dtype=int), dat)
                counts, bin_edges = np.histogram(unbinned, bins=num_bins)
                centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                cache_dati[nome_file] = (centers, counts)
            except Exception as e:
                print(f"Errore lettura file {nome_file}: {e}")
                continue
        
        centers, counts = cache_dati[nome_file]
        
        # Fit
        mu, sigma, err, _ = esegui_fit(centers, counts, sorgente, bar_printing=bar_printing, visualizzare=fits)
        
        if mu is not None:
            punti_ch = np.concatenate((punti_ch, mu))
            punti_E = np.concatenate((punti_E, sorgente['energia']))
            errori_ch = np.concatenate((errori_ch, err))

            punti_ch_sigma = np.concatenate((punti_ch_sigma, sigma))

            gattuso = np.empty(len(mu),dtype="U50")
            gattuso[0] = sorgente['nome']
            nomi_souce = np.concatenate((nomi_souce, gattuso))

            if sorgente.get('fit or not') is not None:
                fit_or_not = np.concatenate((fit_or_not, sorgente['fit or not']))
            else: 
                fit_or_not = np.concatenate((fit_or_not, np.ones(len(mu))))


    # --- 5. RISULTATI FINALI ---

    if len(punti_ch) < 2:
        print("\nERRORE: Non ho trovato abbastanza picchi validi per calibrare.")
        exit()

    elon = fit_or_not == 1
    x_val = punti_ch[elon]
    y_val = punti_E[elon]
    errori_ch= errori_ch[elon]
    nomi_souce = nomi_souce[elon]

    # Interpolazione con una parabola 

    coeff = np.polyfit(x_val, y_val, deg=2)
    A, B, C = coeff
    p = np.poly1d(coeff)
    
    y_fit = p(x_val)
    residuals = y_val - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_val - np.mean(y_val))**2)

    r2 = 1 - ss_res/ss_tot

    # Fit per la risoluzione

    #converto le larghezze da ch -> eV
    punti_E_sigma = abs(2 * A * punti_ch[elon] + B) * punti_ch_sigma[elon]

    #interpolo con una parabola
    coeff_risol = np.polyfit(y_fit, punti_E_sigma**2, deg=2)
    s = np.poly1d(coeff_risol)

    sigma_fit2 = s(y_fit)
    residuals_risol = punti_E_sigma**2 - sigma_fit2
    ss_res_risol = np.sum(residuals_risol**2)
    ss_tot_risol = np.sum((punti_ch_sigma**2 - np.mean(punti_ch_sigma**2))**2)

    s2 = 1 - ss_res_risol/ss_tot_risol

    
    if not bar_visualizzare:
        print("\n" + "="*40)
        print(" RISULTATI CALIBRAZIONE COMPLETA")
        print("="*40)
        for c, e, n in zip(x_val, y_val, nomi_souce):
            print(f" Canale {c:.2f} -> {e:.2f} keV    {n}")

        print("-" * 40)
        print(f"Termin di secondo grado: {A:.5f} KeV/Canale^2")
        print(f"Termin di primo grado: {B:.5f} KeV/Canale")
        print(f"Offset: {C:.3f} keV")
        print(f"Linearità (R^2):       {r2:.6f}")
        print("-" * 40)
        print(f"FORMULA: E [keV] = {A:.5f} * Canale^2 + ({B:.3f}) * Canale + ({C:.3f})")

        # Grafico Finale
        fig_final, ax_top = plt.subplots(figsize=(6, 4))


        ax_top.errorbar(x_val, y_val, xerr=errori_ch, fmt='o', color='blue', label='Punti Sperimentali')
        x_line = np.linspace(0, max(x_val)*1.1, 100)
        ax_top.plot(x_line, p(x_line), 'r-', label=f'Fit ($R^2$={r2:.5f})')

        ax_top.set_ylabel("Energia [KeV]")
        ax_top.set_title(f"Curva di Calibrazione (5 Punti)\nE = {A:.4f}C^2 + ({B:.2f})*C + ({C:.2f})")
        ax_top.grid(True, linestyle='--', alpha=0.5)
        ax_top.legend()
        
        divider = make_axes_locatable(ax_top)
        ax_bottom = divider.append_axes("bottom", size="25%", pad=0.0, sharex=ax_top)
        
        res = (y_val - p(x_val))
        ax_bottom.errorbar(x_val, res, fmt='o', color='blue', label='Residui')
        ax_bottom.axhline(0, linestyle='--')
        ax_bottom.grid(True, linestyle='--', alpha=0.5)
        ax_bottom.set_xlabel("Canale [u.a.]")
        ax_bottom.set_ylabel("Residui")

        ax_top.tick_params(axis='x', labelbottom=False)
        ax_bottom.tick_params(top=False)

        plt.show()

        fig_final, ax_top = plt.subplots(figsize=(6, 4))


        ax_top.errorbar(y_fit, punti_E_sigma, fmt='o', color='blue', label='Punti Sperimentali')
        x_line = np.linspace(0, max(y_fit)*1.1, 100)
        ax_top.plot(x_line, np.sqrt(s(x_line)), 'r-', label=f'Fit ($R^2$={s2:.5f})')

        ax_top.set_ylabel("Larghezze [KeV]")
        ax_top.set_title(f"Curva di Risoluzione (5 Punti)\nSigma^2 = {coeff_risol[0]:.4f}E^2 + ({coeff_risol[1]:.2f})*C + ({coeff_risol[2]:.2f})")
        ax_top.grid(True, linestyle='--', alpha=0.5)
        ax_top.legend()
        
        divider = make_axes_locatable(ax_top)
        ax_bottom = divider.append_axes("bottom", size="25%", pad=0.0, sharex=ax_top)
        
        res = (punti_E_sigma - np.sqrt(s(y_fit)))
        ax_bottom.errorbar(y_fit, res, fmt='o', color='blue', label='Residui')
        ax_bottom.axhline(0, linestyle='--')
        ax_bottom.grid(True, linestyle='--', alpha=0.5)
        ax_bottom.set_xlabel("Energia [KeV]")
        ax_bottom.set_ylabel("Residui")

        ax_top.tick_params(axis='x', labelbottom=False)
        ax_bottom.tick_params(top=False)

        plt.show()

    return coeff, r2, coeff_risol, s2






def main():
    """
    Main entry point for the calibration of one day.
    Parses command line arguments and runs the calibration.
    """
    parser = argparse.ArgumentParser(
        description="Run the calibration analysis in a specific date.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        '-d', 
        '--day', 
        type=str, 
        default="20_11_25",
        help=f"Set date from which take data. (Default: 20_11_25)"
    )

    parser.add_argument(
        '-f', 
        '--fits', 
        action='store_true',
        default=False,
        help=f"View all graphs from fits."
    )

    parser.add_argument(
        '-p', 
        '--print', 
        action='store_true',
        default=False,
        help=f"Disabe printing console's messages."
    )

    parser.add_argument(
        '-v', 
        '--view', 
        action='store_true',
        default=False,
        help=f"View final calibration curves."
    )

    args = parser.parse_args()

    # Pass the parsed arguments to your main simulation function
    print(f"Starting calibration of {args.day}.\n")
    print("======================================================")


    calibration(day=args.day, bar_printing=args.print, fits=args.fits, bar_visualizzare=args.view)

if __name__ == "__main__":
    main()
