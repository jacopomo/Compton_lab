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

def unbin(file):
    """
    Dato un file che contiene i conteggi per ogni bin (in una colonna) di un istogramma, 
    restituisce un array dei valori che sono stati binnati.

    file [str]: file dell'istogramma.
    """
    dat = np.loadtxt(file, dtype=int, unpack=True)
    bin_indices = np.arange(dat.size, dtype=int)
    unbinned = np.repeat(bin_indices, dat)

    return unbinned

def cast_value(value, dtype):
    """
    Converte un valore basandodi sul tipo secificato con una stringa.

    value [str]: valore da convertire il type.
    dtype [str]: type in cui va convertito value.
    """
    if value == "na":
        return None
    
    value = value.strip()

    if dtype == 'str':
        return value
    elif dtype == "int":
        return int(value)
    elif dtype == "float":
        return float(value)
    elif dtype == "list":
        return ast.literal_eval(value)
    else:
        return value

def gen_dict(file):
    """
    Genera una lista di dict da un file .txt usando la prima colonna come keywords per ogni disct che rappresenta ogni riga: 
    nel dict della riga ogni valore e' associato al nome della colonna, usato come keyword. La seconda riga del file .txt contiene 
    i type degli elementi nella colonna.

    file [str]: il path del file.txt.
    """
    df = pd.read_csv(file,
                    sep=r"\s+",
                    engine="python",
                    dtype=str)
    df = df.replace({np.nan: None})

    value = {}
    key_col = df.columns[0]

    for i,row in df.iterrows():
        if i == 0:
            continue
        main_key = row[key_col]
        nested = {}

        for col,val in row.items():
            if col == key_col:
                continue
            if val != "na":
                nested[col] = cast_value(val, dtype=df.loc[0,col])
        value[main_key] = nested

    return value




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

    

