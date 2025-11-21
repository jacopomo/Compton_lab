import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- 1. MODELLO MATEMATICO ---
def modello_fit(x, A, mu, sigma, B0, k, B1):
    """
    Gaussiana (Picco) + Esponenziale (Fondo)
    """
    gaussiana = A * np.exp(-0.5 * ((x - mu) / sigma)**2)
    # Aggiungiamo un piccolo epsilon per evitare errori log con k*x
    fondo = B0 * np.exp(-k * x) + B1
    return gaussiana + fondo

# --- 2. CARICAMENTO DATI ---
file_name = input("Nome file Cesio (137Cs) -> ")
file = r"Dati/" + file_name
resize_factor = int(input("Resize factor -> "))
num_bins = int(8192 / resize_factor)

dat = np.loadtxt(file, dtype=int, unpack=True)
unbinned = np.repeat(np.arange(dat.size, dtype=int), dat)
counts, bin_edges = np.histogram(unbinned, bins=num_bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# --- 3. VISUALIZZAZIONE PRELIMINARE PER SCELTA FINESTRA ---
plt.figure(figsize=(10,4))
plt.step(bin_centers, counts, where='mid')
plt.title("Spettro Completo - Individua la finestra del Picco")
plt.show()

# --- 4. INPUT PARAMETRI FINESTRA ---
print("\n--- Configurazione Fit ---")
low_ch = float(input("Canale INIZIO finestra fit -> "))
high_ch = float(input("Canale FINE finestra fit -> "))

# Selezione dati
mask = (bin_centers >= low_ch) & (bin_centers <= high_ch)
x_fit = bin_centers[mask]
y_fit = counts[mask]

if len(x_fit) == 0:
    print("Finestra vuota!")
    exit()

# --- 5. STIMA AUTOMATICA (GUESS) ---
mu_auto = x_fit[np.argmax(y_fit)]      # Il picco è il massimo
A_auto = np.max(y_fit) - np.min(y_fit) # Altezza relativa
sigma_auto = (high_ch - low_ch) / 10.0 # Stima larghezza
B0_auto = y_fit[0]                     # Valore a sinistra (inizio exp)
B1_auto = y_fit[-1]                    # Valore a destra (offset)
# Stima k: assumiamo che il fondo scenda di un fattore 2 nella finestra
# B_start = B0 * e^(-k*x_start) -> k approssimato
k_auto = 0.001 

print("\n--- PARAMETRI INIZIALI (GUESS) ---")
print(f"1. Ampiezza (A)  [Auto: {A_auto:.0f}]")
print(f"2. Centro (mu)   [Auto: {mu_auto:.1f}]")
print(f"3. Sigma         [Auto: {sigma_auto:.1f}]")
print(f"4. Fondo Amp (B0)[Auto: {B0_auto:.0f}]")
print(f"5. Decadim. (k)  [Auto: {k_auto:.5f}]")
print(f"6. Fondo Off (B1)[Auto: {B1_auto:.0f}]")

scelta = input("\nVuoi modificare manualmente i parametri iniziali? (s/n) -> ")

if scelta.lower() == 's':
    p0 = [
        float(input(f"Inserisci A (default {A_auto:.0f}): ")),
        float(input(f"Inserisci mu (default {mu_auto:.1f}): ")),
        float(input(f"Inserisci sigma (default {sigma_auto:.1f}): ")),
        float(input(f"Inserisci B0 (default {B0_auto:.0f}): ")),
        float(input(f"Inserisci k (default {k_auto:.5f}): ")),
        float(input(f"Inserisci B1 (default {B1_auto:.0f}): "))
    ]
else:
    p0 = [A_auto, mu_auto, sigma_auto, B0_auto, k_auto, B1_auto]

# --- 6. ESECUZIONE FIT ---
try:
    # Bounds: impediamo valori negativi per A, sigma, B0, k
    # (0, np.inf) per tutti tranne mu e B1 che lasciamo liberi o vincolati se serve
    bounds_min = [0, low_ch, 0, 0, 0, -np.inf]
    bounds_max = [np.inf, high_ch, np.inf, np.inf, 1.0, np.inf]
    
    popt, pcov = curve_fit(modello_fit, x_fit, y_fit, p0=p0, bounds=(bounds_min, bounds_max), maxfev=50000)
except Exception as e:
    print(f"\nERRORE NEL FIT: {e}")
    print("Prova a cambiare la finestra o i parametri iniziali.")
    exit()

# Risultati
A, mu, sigma, B0, k, B1 = popt
perr = np.sqrt(np.diag(pcov))

print("\n" + "="*30)
print(f"RISULTATO FIT (Chi-Quadro ridotto non calcolato)")
print(f"Centro Picco (mu): {mu:.2f} +/- {perr[1]:.2f}")
print(f"Sigma:             {sigma:.2f}")
print(f"Costante k fondo:  {k:.5f}")
print("="*30)

# --- 7. CALIBRAZIONE (1 PUNTO) ---
E_cesio = 661.7
m = E_cesio / mu
q = 0 # Assunto

print(f"\n--- CALIBRAZIONE (Assumendo Offset=0) ---")
print(f"Pendenza m: {m:.5f} keV/canale")
print(f"Formula: E = {m:.5f} * Ch")

# --- 8. VISUALIZZAZIONE ---
plt.figure(figsize=(10,6))
# Dati
plt.plot(x_fit, y_fit, 'b.', label='Dati (Finestra)', alpha=0.6)
# Fit Totale
x_plot = np.linspace(min(x_fit), max(x_fit), 1000)
plt.plot(x_plot, modello_fit(x_plot, *popt), 'r-', linewidth=2, label=f'Fit Totale')
# Componenti
plt.plot(x_plot, B0*np.exp(-k*x_plot) + B1, 'g--', label='Fondo Esponenziale')
plt.plot(x_plot, A*np.exp(-0.5*((x_plot-mu)/sigma)**2) + (B0*np.exp(-k*x_plot) + B1), 'k:', alpha=0.3, label='Gaussiana')

plt.title(f"Fit Cesio-137\nPicco @ ch {mu:.2f}")
plt.xlabel("Canale")
plt.ylabel("Conteggi")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()