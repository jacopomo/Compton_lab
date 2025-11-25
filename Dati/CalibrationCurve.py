import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import json
import os

# --- 1. MODELLO MATEMATICO ---
def modello_gauss_exp(x, A, mu, sigma, B0, k, B1):
    """Gaussiana (Picco) + Fondo Esponenziale"""
    # Protezione per evitare overflow dell'esponenziale
    k = np.abs(k)
    gauss = A * np.exp(-0.5 * ((x - mu) / sigma)**2)
    fondo = B0 * np.exp(-k * x) + B1 
    return gauss + fondo

# --- 2. FUNZIONE DI FIT ---
def esegui_fit(bin_centers, counts, config_sorgente, printing=False, visualizzare=False):
    nome = config_sorgente['nome']
    energia = config_sorgente['energia']
    finestra = config_sorgente['finestra'] # [low, high]
    manual_guess = config_sorgente.get('guess') # Opzionale

    if printing:
        print(f"\nAnalisi: {nome} ({energia} keV)...")

    # Selezione Finestra
    low, high = finestra
    mask = (bin_centers >= low) & (bin_centers <= high)
    x_win = bin_centers[mask]
    y_win = counts[mask]

    # Controllo preliminare dati
    if len(x_win) < 6:
        print(f"  --> ERRORE: Finestra [{low}-{high}] vuota o troppo stretta!")
        return None, None

    # Stima Parametri Iniziali (Guess)
    if manual_guess is not None and len(manual_guess) == 6:
        p0 = manual_guess
        if printing:
            print("  --> Uso parametri manuali dal JSON.")
    else:
        # Logica automatica intelligente
        mu_g = x_win[np.argmax(y_win)]       # Picco = Massimo
        bg_min = np.min(y_win)
        bg_max = np.max(y_win)
        A_g = bg_max - bg_min                # Altezza relativa
        sigma_g = (high - low) / 6.0         # Larghezza generica
        B0_g = y_win[0]                      # Valore a sinistra
        B1_g = 0.0                           # Offset
        k_g = 0.0005                         # Pendenza lieve
        
        # Correzione specifica per Americio (spesso il fondo sale a sinistra violentemente)
        if energia < 100: 
             k_g = 0.005 # Pendenza più forte per basse energie
        
        p0 = [A_g, mu_g, sigma_g, B0_g, k_g, B1_g]
        if printing:
            print("  --> Uso parametri automatici.")

    # Esecuzione Fit
    try:
        # Bounds per evitare risultati fisicamente impossibili (es. ampiezza negativa)
        # A, mu, sigma, B0, k, B1
        bounds_min = [0, low, 0, 0, -np.inf, -np.inf]
        bounds_max = [np.inf, high, np.inf, np.inf, 1, np.inf]
        
        popt, pcov = curve_fit(modello_gauss_exp, x_win, y_win, p0=p0, bounds=(bounds_min, bounds_max), maxfev=20000)
        err_mu = np.sqrt(np.diag(pcov))[1]
        
        if visualizzare:
            # Visualizzazione risultato fit
            plt.figure(figsize=(6, 4))
            plt.step(bin_centers, counts, where='mid', color='lightgray', label='Spettro intero')
            plt.plot(x_win, y_win, 'b.', label='Dati Finestra')
            
            x_plot = np.linspace(min(x_win), max(x_win), 500)
            plt.plot(x_plot, modello_gauss_exp(x_plot, *popt), 'r-', linewidth=2, label=f'Fit (mu={popt[1]:.2f})')
            
            plt.title(f"Fit: {nome}")
            plt.xlabel("Canale")
            plt.ylabel("Conteggi")
            plt.xlim(low*0.8, high*1.2) # Zoom
            plt.legend()
            plt.show()

        if printing:
            print(f"  --> OK! Picco trovato a canale {popt[1]:.2f} +/- {err_mu:.2f}")
        return popt[1], err_mu

    except Exception as e:
        print(f"  --> FIT FALLITO: {e}")
        print("      Suggerimento: controlla la finestra nel JSON.")
        return None, None

# --- 3. LETTURA DATI E CONFIGURAZIONE ---
file_config = "config_calibration.json"

if not os.path.exists(file_config):
    print(f"ERRORE: Devi creare il file '{file_config}' nella cartella!")
    exit()

with open(file_config, 'r') as f:
    config_globale = json.load(f)

resize = config_globale['resize_factor']
lista_sorgenti = config_globale['sorgenti']
num_bins = int(8192 / resize)

print(f"--- AVVIO CALIBRAZIONE AUTOMATICA (Resize: {resize}) ---")

# --- 4. CICLO DI ANALISI ---
punti_ch = []
punti_E = []
errori_ch = []
cache_dati = {} # Per non ricaricare i file se usati più volte

for sorgente in lista_sorgenti:
    nome_file = sorgente['file']
    
    # Caricamento (solo se non già in memoria)
    if nome_file not in cache_dati:
        path = r"Dati/Calibration/" + nome_file
        if not os.path.exists(path):
            print(f"ATTENZIONE: File '{path}' non trovato. Salto {sorgente['nome']}.")
            continue
        
        try:
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
    mu, err = esegui_fit(centers, counts, sorgente, printing=True, visualizzare=True)
    
    if mu is not None:
        punti_ch.append(mu)
        punti_E.append(sorgente['energia'])
        errori_ch.append(err)

# --- 5. RISULTATI FINALI ---

if len(punti_ch) < 2:
    print("\nERRORE: Non ho trovato abbastanza picchi validi per calibrare.")
    exit()

x_val = np.array(punti_ch)
y_val = np.array(punti_E)

# Regressione Lineare
res = linregress(x_val, y_val)
m, q, r2 = res.slope, res.intercept, res.rvalue**2

print("\n" + "="*40)
print(" RISULTATI CALIBRAZIONE COMPLETA")
print("="*40)
for c, e in zip(x_val, y_val):
    print(f"  Canale {c:.2f} -> {e:.2f} keV")

print("-" * 40)
print(f"Pendenza (Guadagno) m: {m:.5f} keV/Canale")
print(f"Offset (Intercetta) q: {q:.3f} keV")
print(f"Linearità (R^2):       {r2:.6f}")
print("-" * 40)
print(f"FORMULA: E [keV] = {m:.5f} * Canale + ({q:.3f})")

# Grafico Finale
plt.figure(figsize=(8, 6))
plt.errorbar(x_val, y_val, xerr=errori_ch, fmt='o', color='blue', label='Punti Sperimentali')
x_line = np.linspace(0, max(x_val)*1.1, 100)
plt.plot(x_line, m*x_line + q, 'r-', label=f'Fit Lineare ($R^2$={r2:.5f})')

plt.xlabel("Canale")
plt.ylabel("Energia (keV)")
plt.title(f"Curva di Calibrazione (5 Punti)\nE = {m:.4f}C + {q:.2f}")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()