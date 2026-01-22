import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd # Utilizzeremo pandas per un salvataggio CSV più pulito

from CalibrationCurve import calibration

def find_paths_for_angle(angle_str: str) -> tuple[Path, Path]:
    """
    Trova i percorsi per il file dati e la cartella di calibrazione per un dato angolo.
    
    Restituisce:
        Una tupla (data_file_path: Path, calibration_folder_path: Path)
    """
    
    # Calcolo del percorso base (presupponendo che 'Analisi' sia accanto a 'Dati')
    base_path = Path(__file__).parent.parent / 'Dati' / 'Measures' / 'Angles'
    angle_name = f"{angle_str}deg"
        
    file_pattern = f"{angle_name}_*.dat"
    data_files = list(base_path.glob(file_pattern))
    
    if not data_files:
        raise ValueError(f"Errore: Nessun file dati trovato con pattern '{file_pattern}' in {base_path.resolve()}")
    
    if len(data_files) > 1:
        print(f"Avviso: Trovati più file dati per {angle_str}deg. Verrà usato il primo trovato.")

    data_file_path = data_files[0]

    filename = data_file_path.stem                      # e.g. "20deg_251125"
    date = filename.split("_", 1)[1]                    # "251125"
    date_str = f"{date[:2]}_{date[2:4]}_{date[4:6]}"    # "25_11_25"
    calibration_folder_path = Path(__file__).parent.parent / 'Dati' / 'Calibration' / date_str

    if not calibration_folder_path.is_dir():
        raise ValueError(f"Errore: Directory Calibration non trovata in: {calibration_folder_path.resolve()}")

    return data_file_path, calibration_folder_path


def save_energies_as_csv(energies: np.ndarray, angle_str: str):
    """
    Salva l'array di energie in un file CSV nella cartella Spettri_calibrati.
    """
    # Il percorso della cartella di destinazione si trova nella stessa directory dello script (Analisi)
    output_dir = Path(__file__).parent / 'Analisi_angoli' / 'Spettri_calibrati'
    output_dir.mkdir(parents=True, exist_ok=True) # Crea la cartella se non esiste

    file_name = f"{angle_str}deg_calibrato.csv"
    output_path = output_dir / file_name

    # Utilizza pandas per salvare l'array come una singola colonna senza header o indice
    # in modo che il file .csv contenga solo i valori delle energie.
    series = pd.Series(energies, name="Energia [keV]")
    series.to_csv(output_path, index=False, header=True)

    print(f"\nFile salvato con successo in: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Process data file for a specific angle.")
    parser.add_argument('angle', type=str, help="The measurement angle to process (e.g., 0, 10, 30).")
    args = parser.parse_args()
    angle_input = args.angle

    try:
        data_file_path, cal_folder_path = find_paths_for_angle(angle_input)
        
        print(f"Trovato file dati: {data_file_path.resolve()}")
        print(f"Trovata cartella calibrazione: {cal_folder_path.resolve()}")

        # 1. Carica i dati grezzi (conteggi per canale)
        dat = np.loadtxt(data_file_path, dtype=int, unpack=True)
        
        # 2. Converte i conteggi in una lista "unbinned" di canali individuali
        bin_indices = np.arange(dat.size, dtype=int)
        unbinned_channels = np.repeat(bin_indices, dat)
        
        # 3. Ottieni i coefficienti di calibrazione (m, q)
        # Assicurati che la funzione calibration restituisca esattamente m, q e qualcos'altro
        m, q, _ = calibration(cal_folder_path, vis=False) 
        
        # 4. Converti i canali in energie (E = m * canali + q, o come specificato dalla tua formula)
        # La tua formula precedente era: (unbinned_channels * q) * m; ho assunto che m e q siano i coefficienti di una retta
        # Modifica la riga seguente per adattarla alla logica esatta della tua funzione calibration()
        unbinned_energies = (unbinned_channels * m) + q
        
        # 5. Salva i risultati in un CSV
        save_energies_as_csv(unbinned_energies, angle_input)

    except ValueError as e:
        print(e)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Un componente del percorso non è stato trovato.")
        sys.exit(1)
    except Exception as e:
        print(f"Si è verificato un errore inaspettato: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
