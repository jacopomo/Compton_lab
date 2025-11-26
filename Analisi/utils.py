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
    A [float]: A * sqrt(2) * sigma corrisponde all'integrale su [-inf,inf]
    """
    return A * np.exp(-0.5 * ((x-mu)/sigma)**2)

def gauss_exp(x, A, mu, sigma, B0, k, B1):
    """
    Somma di una gaussiana, di un esponenziale e di una costante.
    
    x [float]: Variabile indipendente.
    mu [float]: media della gaussiana.
    sigma [float]: larghezza della gaussiana.
    A [float]: A * sqrt(2) * sigma corrisponde all'integrale su [-inf,inf] della gaussiana
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
    A* [float]: A * sqrt(2) * sigma corrisponde all'integrale su [-inf,inf] della gaussiana
    k [float]: coefficiente della caduta esponenziale.
    B0 [float]: fattore di normalizzazione, A/k corrisponde all'integrale su [0,inf] dell'esponenziale.
    B1 [float]: Costante di offset.
    """

    g1 = gauss(x, mu1, sigma1, A1)
    g2 = gauss(x, mu2, sigma2, A2)
    fondo = expo(x, k, B0) + B1 
    return g1 + g2 + fondo

    

