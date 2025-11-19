import numpy as np
import glob as gl


def searchfiles(relative_path, ext):
    """
    Cerca in un una directory tutti i file con l'estensione indicata 'ext'.
    
    relative_path [str]: la directory in cui cercare.
    ext [str]: l'estensione da cercare.
     """
    key = relative_path + "/*." + ext
    files = gl.glob(key)

    return files

def unbin(file):
    """
    Dato un file che contiene i conteggi per ogni bin (in una colonna) di un istogramma, 
    restituisce un array dei valori che sono stati binnati.

    file [str]: file dell'istogramma.
    """
    dat = np.loadtxt(file, dtype=int, unpack=True)

    unbinned = np.array([])
    for i,mis in enumerate(dat):
        if mis != 0:
            new = np.concatenate((new, np.full(mis, i)))

    return unbinned



def expo(x, lam, A):
    """
    Funzine del tipo A * exp(-lambda x)

    x [float]: Variabile indipendente.
    lambda [float]: coefficiente della caduta esponenziale.
    A [float]: fattore di normalizzazione, A/lambda corrisponde all'integrale su [0,inf].
    """
    return A * np.exp(-1 * lam * x)

def gauss(x, mu, sigma, A):
    """
    Funzione gaussiana non normalizzata a 1.
    
    x [float]: Variabile indipendente.
    mu [float]: media.
    sigma [float]: larghezza.
    A [float]: A * sqrt(2) * sigma corrisponde all'integrale su [-inf,inf]
    """
    return A * np.exp(-0.5 * ((x-mu)/sigma)**2)

    

