import numpy as np
import glob as gl
import pandas as pd
import ast


def searchfiles(relative_path, ext):
    """
    Cerca in un una directory tutti i file con l'estensione indicata 'ext'.
    
    relative_path [str]: la directory in cui cercare.
    ext [str]: l'estensione da cercare.
     """
    key = relative_path + "/*." + ext
    files = gl.glob(key)

    return files



def expo(x, k, A):
    """
    Funzine del tipo A * exp(-lambda x)

    x [float]: Variabile indipendente.
    k [float]: coefficiente della caduta esponenziale.
    A [float]: fattore di normalizzazione, A/k corrisponde all'integrale su [0,inf].
    """
    return A * np.exp(-1 * k * x)

def gauss(x, mu, sigma, A):
    """
    Funzione gaussiana non normalizzata a 1.
    
    x [float]: Variabile indipendente.
    mu [float]: media.
    sigma [float]: larghezza.
    A [float]: corrisponde all'integrale su [-inf,inf].
    """
    return A / (np.sqrt(2*pi) * sigma) * np.exp(-0.5 * ((x-mu)/sigma)**2)

def emg(x, mu, sigma, tau, A):
    """
    Funzione EMG, Exponential Modified Gaussian.

    x [float]: Variabile indipendente.
    mu [float]: media della gaussiana.
    sigma [float]: larghezza della gaussiana.
    tau [float]: parametro della coda esponenziale, tau>0 per coda verso canali piu' bassi.
    A [float]: corrisponde all'integrale su [-inf, inf]. 
    """
    x = np.asarray(x)
    arg1 = (sigma**2) / (2.0 * tau**2) - (x - mu) / tau
    arg2 = (sigma / tau - (x - mu) / sigma) / np.sqrt(2.0)
    return (A / (2.0 * tau)) * np.exp(arg1) * np.erfc(arg2)




def gauss_exp(x, A, mu, sigma, B0, k, B1):
    """
    Somma di una gaussiana, di un esponenziale e di una costante.
    
    x [float]: Variabile indipendente.
    mu [float]: media della gaussiana.
    sigma [float]: larghezza della gaussiana.
    A [float]: corrisponde all'integrale su [-inf,inf] della gaussiana.
    k [float]: coefficiente della caduta esponenziale.
    B0 [float]: fattore di normalizzazione, A/k corrisponde all'integrale su [0,inf] dell'esponenziale.
    B1 [float]: Costante di offset.
    """

    g = gauss(x, mu, sigma, A)
    fondo = expo(x, k, B0) + B1 
    return g + fondo

def double_gauss_exp(x, A1, mu1, sigma1, A2, mu2, sigma2, B0, k, B1):
    """
    Somma di due gaussiane, di un esponenziale e di una costante.
    
    x [float]: Variabile indipendente.
    mu* [float]: media della gaussiana.
    sigma* [float]: larghezza della gaussiana.
    A* [float]: corrisponde all'integrale su [-inf,inf] della gaussiana
    k [float]: coefficiente della caduta esponenziale.
    B0 [float]: fattore di normalizzazione, A/k corrisponde all'integrale su [0,inf] dell'esponenziale.
    B1 [float]: Costante di offset.
    """

    g1 = gauss(x, mu1, sigma1, A1)
    g2 = gauss(x, mu2, sigma2, A2)
    fondo = expo(x, k, B0) + B1 
    return g1 + g2 + fondo

def asym_gauss_exp(x, A1, mu, sigma, A2, delta, c, B0, k, B1):
    """
    Somma di due gaussiane, di un esponenziale e di una costante.
    
    x [float]: Variabile indipendente.
    mu [float]: media della gaussiana.
    sigma [float]: larghezza della gaussiana.
    A* [float]: sigma corrisponde all'integrale su [-inf,inf] della gaussiana
    delta [float]: distanza tra le due gaussiane.
    c [float]: rapporto tra le due larghezze delle due gaussiane.
    k [float]: coefficiente della caduta esponenziale.
    B0 [float]: fattore di normalizzazione, A/k corrisponde all'integrale su [0,inf] dell'esponenziale.
    B1 [float]: Costante di offset.
    """

    g1 = gauss(x, mu, sigma, A1)
    g2 = gauss(x, mu - delta, c * sigma, A2)
    fondo = expo(x, k, B0) + B1 
    return g1 + g2 + fondo

    

