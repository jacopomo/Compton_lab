import numpy as np
import matplotlib.pyplot as plt


file_name = input("Nome file -> ")
file = r"Dati/" + file_name 

resize_factor = int(input("Resize factor dell'istogramma -> "))
bins_tot = 8192
num_bins = int(bins_tot / resize_factor)


dat = np.loadtxt(file, dtype=int, unpack=True)
bin_indices = np.arange(dat.size, dtype=int)
unbinned = np.repeat(bin_indices, dat)



plt.figure()
counts, bin_edges, _ = plt.hist(unbinned, bins=num_bins, histtype='step', visible=False)
plt.close() 

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 # Centri dei Bin (Asse X)

print("\n--- Selezione Finestra di Fit (in Canali ORIGINALI 0-8191) ---")
try:
    low_channel_orig = float(input("Canale iniziale ORIGINALE per il fit -> "))
    high_channel_orig = float(input("Canale finale ORIGINALE per il fit -> "))
except ValueError:
    print("Input non valido. Inserire numeri validi.")
    exit()


fit_indices = np.where((bin_centers >= low_channel_orig) & (bin_centers <= high_channel_orig))

x_fit = bin_centers[fit_indices] 
y_fit = counts[fit_indices]      

# Controllo per array insufficiente (per prevenire TypeError e fit non stabili)
if len(x_fit) < 5:
    print("\nERRORE: La finestra selezionata non contiene un numero sufficiente di punti per un fit di grado 4.")
    exit()


coefficients = np.polyfit(x_fit, y_fit, 4) 
p = np.poly1d(coefficients) #funzione
y_fitted = p(x_fit)

print("\n--- Risultati del Fit (Polinomio di Grado 4) ---")
print(f"Coefficienti [a, b, c, d, e]:\n{coefficients}")
print(f"Polinomio P(x) = {p}")


plt.figure(figsize=(10, 6))
plt.hist(unbinned, bins=num_bins, histtype='step', label='Dati Misurati (Rebinati)', color='gray')
plt.plot(x_fit, y_fit, 'o', markersize=3, label=f'Dati Finestra Fit ({low_channel_orig}-{high_channel_orig})', color='blue')
plt.plot(x_fit, y_fitted, '-', label=f'Fit Polinomiale (Grado 4)', color='red', linewidth=2)
plt.title(f'Fit Polinomiale su Finestra Selezionata\nFile: {file_name}')
plt.xlabel(f'Canale (Rebin factor: {resize_factor})')
plt.ylabel('Conteggi')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()