##### WORK IN PROGRESS!!!!
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.interpolate as interp
from scipy.interpolate import CubicSpline
import time
import os

######## Constanti ########
N_MC = int(5e6) # Num samples (MASSIMO 5e7 SE NON VUOI FAR DIVENTARE IL TUO COMPUTER UN TERMOSIFONE)

### Geometria:
RCOL = 1 # Raggio collimatore [cm]
L = 11 #Lunghezza collimatore [cm]
SAACOLL = np.degrees(np.arctan(2/L)) # Semi-apertura angolare del collimatore [gradi]
RP = 2.5 # Raggio plastico [cm]
LP = 3 # Lunghezza plastico [cm]
DSP = 1.5 # Distanza sorgente - plastico [cm]
DBC = 47 # Distanza bersaglio - cristallo [cm]
RC = 2.54 # Raggio del cristallo [cm]
LC = 5.08 # Lunghezza del cristallo [cm]
SAAC = np.arctan(RC/DBC) # Semi-apertura angolare del cristallo [rad]
FLUSSO = 2258 # Fotoni al secondo

PHI = 20  # Angolo al quale si trova il cristallo [gradi]


### Fisica
ALPHA = 1/137 # Costante di struttura fine
ME = 511 # Massa dell'elettrone [keV]
RE = 2.8179403262e-13 #[cm]
E1, E2 = 1173.240, 1332.508 # Energie dei fotoni

### Config
THETA_MIN, THETA_MAX = 0, np.pi # Theta min e max
THETA_MESH = np.linspace(THETA_MIN, THETA_MAX, 250) # Theta mesh
E_ref = np.linspace(1, 2000, 100)  # 100 bins da 1 keV a 2000 keV
E_ref = np.concatenate(([E1, E2], E_ref))  # Energie importanti

ESOGLIA_C = 550 # Soglia del cristallo [keV]
EBINMAX = 2000 # Massimo del binning [keV]
NBINS = 80

MAX_CML_TRIES = 500  # Maximum times to re-sample cml for escaping photons

np.random.seed(42) # Seed

######## Classi ########
class Materiale:
    def __init__(self, formula, density):
        """""
        formula: str
        density: float [g/cm^3]
        """""
        self.formula = formula
        self.density = density
        self._splines_built = False

    def _build_splines(self):
        """Metodo privato, leggi il file una volta"""
        base_dir = os.path.dirname(__file__)
        filepath = os.path.join(base_dir, "Dati_materiali", f"{self.formula}.txt")
        ph_E, sigma_c, sigma_pe, sigma_tot = np.loadtxt(filepath, skiprows=2, unpack=True)
        ph_E = ph_E * 1000  # MeV to keV

        self.sigma_c_spline  = CubicSpline(ph_E, sigma_c)
        self.sigma_pe_spline = CubicSpline(ph_E, sigma_pe)
        self.sigma_tot_spline = CubicSpline(ph_E, sigma_tot)
        self._splines_built = True

    def sigmas(self):
        if not self._splines_built:
            self._build_splines()
        return {"compton": self.sigma_c_spline, "fotoelettrico": self.sigma_pe_spline, "totale": self.sigma_tot_spline}

    def cml(self, E):
        """ Trova il cammino libero medio e lo campiona

        Parametri: 
        E: Energia in keV del fotone
        Returns: Cammino in cm
        """
        E = np.atleast_1d(E)
        sigma_pe = self.sigmas()["fotoelettrico"](E)
        sigma_c = self.sigmas()["compton"](E)

        clm_pe = 1/(self.density*sigma_pe)
        clm_c = 1/(self.density*sigma_c)

        L_pe = -clm_pe*np.log(np.random.uniform(0,1, E.shape))
        L_c = -clm_c*np.log(np.random.uniform(0,1, E.shape))

        mask = L_pe < L_c
        L = np.where(mask, L_pe, L_c)
        interaction = np.where(mask, "Fotoelettrico", "Compton")

        return L, interaction

class Superficie:
    def __init__(self, raggio, centro=(0,0,0), angolo=0): # Passa gradi
        self.centro = np.array(centro)
        self.angolo = np.radians(angolo)
        self.raggio = raggio

    def normal(self):
        if np.linalg.norm(self.centro)==0:
            normal = np.array([0,0,1])
        else:
            normal = self.centro / np.linalg.norm(self.centro)
        return normal
    
    def pos_sul_piano_unif(self, n, debug_graph=False):
        thetas = np.random.uniform(0,2*np.pi, n)
        rs = self.raggio * np.sqrt(np.random.uniform(0,1,n))
        fx = self.centro[0] + (rs*np.sin(thetas))
        fy = self.centro[1] + (rs*np.cos(thetas)*np.cos(self.angolo))
        fz = self.centro[2] + (rs*np.cos(thetas)*np.sin(self.angolo)) 
        if debug_graph:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(fx,fy,fz) 
            plt.show()
        return fx, fy, fz

class Volume:
    def __init__(self, materiale, raggio, lunghezza, centro_sup_vicina=(0,0,0), angolo=0): # Passa gradi
        self.centro_sup_vicina = np.array(centro_sup_vicina)
        self.angolo = np.radians(angolo)
        self.raggio = raggio
        self.lunghezza = lunghezza
        self.materiale = materiale
    
    def cml(self, E):
        return self.materiale.cml(E)

class Fotone:
    def __init__(self, energia, px, py, pz, phi, psi): #Passa gradi phi psi
        self.energia = energia
        self.px = px
        self.py = py
        self.pz = pz
        self.phi = np.radians(phi)
        self.psi = np.radians(psi)


    def calcola_int(self, superficie, debug_graph=False):
        p = np.stack((self.px, self.py, self.pz), axis=-1)
        phi, psi = self.phi, self.psi
        centro = superficie.centro
        normal = superficie.normal()
        
        dx, dy, dz = np.sin(psi), np.sin(phi)*np.cos(psi), np.cos(phi)*np.cos(psi)
        d = np.stack((dx, dy, dz), axis=-1)
        
        num, denom = (centro-p) @ normal, np.sum(d * normal, axis=-1)

        parallel = np.isclose(denom, 0.0, atol=1e-8)
        t = np.full_like(denom, np.nan, dtype=float)

        valid = ~parallel
        t[valid] = num[valid] / denom[valid]
        
        forward = (~np.isnan(t)) & (t >= -1e-8)
        if not np.any(forward):
            return None, None, None, None, None, None, None
    
        pts_all = p + np.expand_dims(t, axis=-1) * d
        idx_forward = np.where(forward)[0]
        pts = pts_all[forward]
        phi_f = phi[forward]
        psi_f = psi[forward]

        d2 = np.sum((pts - centro)**2, axis=1)
        mask = d2 < superficie.raggio**2
        if not np.any(mask):
            return None, None, None, None, None, None, None

        xs, ys, zs = pts[mask].T
        phi_sel = phi_f[mask]
        psi_sel = psi_f[mask]
        # select corresponding original indices to index energies
        selected_indices = idx_forward[mask]
        E_sel = self.energia[selected_indices]

        if debug_graph:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(xs,ys,zs)
            plt.show()

        return xs, ys, zs, np.degrees(phi_sel), np.degrees(psi_sel), E_sel

    def isinside(self, x,y,z, vol):
        """ Vede se il fotone sta dentro un solido
        Mi raccomando, coordinate x,y,z con z l'asse del prisma e l'origine nel centro della sup vicina, in cm
        """

        r = np.sqrt(x**2+y**2)
        mask = (r < vol.raggio) & (z < vol.lunghezza) & (z > -1e-9)
        if mask.shape == ():
            return bool(mask)
        return mask

    def scatter_inside(self, volume, debug_graph=False, pause_time=0.6, debug_save=False, save_gif_path=None, debug_slider=False):

        c = volume.centro_sup_vicina
        a = volume.angolo

        E = self.energia.copy()
        E_depositata = np.zeros_like(E)

        phi, psi = self.phi, self.psi
        x,y,z = self.px-c[0], self.py-c[1], self.pz-c[2]
        y,z = ((y*np.cos(a))-(z*np.sin(a))), ((y*np.sin(a))+(z*np.cos(a))) # Posizione nelle nuove coordinate del sistema cartesiano
        p = np.stack((x, y, z), axis=-1)

        phi, psi = phi-a, psi
        dx, dy, dz = np.sin(psi), np.sin(phi)*np.cos(psi), np.cos(phi)*np.cos(psi)
        d = np.stack((dx, dy, dz), axis=-1)

        active = np.ones(len(E), dtype=bool) # Fotoni "attivi"
        iteration = 0
        frames = None
        # If user requests saving frames or a slider, ensure debug_graph is enabled
        if (debug_save or debug_slider) and not debug_graph:
            print('debug_save/debug_slider requested: enabling debug_graph automatically')
            debug_graph = True

        # For debugging, prepare a figure; we'll plot initial positions and then
        # update the plot per iteration showing only active photons that are
        # inside the volume. Optionally capture frames to save an animation or
        # display with a slider.
        if debug_graph:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(projection='3d')
            ax.set_title(f"Iteration {iteration}: initial face (active photons)")
            ax.set_xlabel('x [cm]')
            ax.set_ylabel('y [cm]')
            ax.set_zlabel('z [cm]')
            # Plot initial positions (face)
            if np.any(active):
                pts0 = p[active]
                ax.scatter(pts0[:,0], pts0[:,1], pts0[:,2], c='k', marker='o', s=10, alpha=0.6, label=f'iter {iteration}')
                ax.legend()
                plt.draw(); plt.pause(pause_time)
            ax.view_init(elev=30, azim=40)
            # Pre-compute static axis limits from volume geometry in the local
            # coordinate system (p is in coordinates relative to volume's
            # surface): x,y limited by radius, z limited by [0, lunghezza].
            if debug_graph:
                r = volume.raggio
                ax.set_xlim(-r, r)
                ax.set_ylim(-r, r)
                ax.set_zlim(0, volume.lunghezza)
            frames = []
            if debug_save:
                # Capture the initial frame
                fig.canvas.draw()
                w,h = fig.canvas.get_width_height()
                try:
                    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))
                except Exception:
                    arr = np.asarray(fig.canvas.buffer_rgba())
                    if arr.ndim == 3 and arr.shape[2] == 4:
                        img = arr[:, :, :3].copy()
                    else:
                        img = arr.copy()
                frames.append(img.copy())
        while np.any(active):
            L, tipo = volume.cml(E[active])
            new_p = p[active] + d[active] * L[:, None]
            inside = self.isinside(new_p[:,0], new_p[:,1], new_p[:,2], volume)

            # If a photon escapes (inside == False) we try regenerating the cml
            # up to MAX_CML_TRIES times trying to obtain a cml that leads to
            # an interaction inside the volume.
            if not np.all(inside):
                # Work with active-local arrays to retry only those that escaped
                n_active = len(L)
                attempts = np.zeros(n_active, dtype=int)
                still_escaped = ~inside
                # Retry up to MAX_CML_TRIES times
                while np.any(still_escaped) and np.any(attempts < MAX_CML_TRIES):
                    to_retry = still_escaped & (attempts < MAX_CML_TRIES)
                    if not np.any(to_retry):
                        break
                    # Resample cml for the subset
                    L_new, tipo_new = volume.cml(E[active][to_retry])
                    L[to_retry] = L_new
                    tipo[to_retry] = tipo_new
                    new_p[to_retry] = p[active][to_retry] + d[active][to_retry] * L[to_retry][:, None]
                    inside[to_retry] = self.isinside(new_p[to_retry,0], new_p[to_retry,1], new_p[to_retry,2], volume)
                    # Update the escape mask and increments attempts only where we retried
                    still_escaped = ~inside
                    attempts[to_retry] += 1
            if debug_graph:
                inside_local = inside
                ax.cla()
                # Keep axes static across frames
                r = volume.raggio
                ax.set_xlim(-r, r)
                ax.set_ylim(-r, r)
                ax.set_zlim(0, volume.lunghezza)
                if np.any(inside_local):
                    ax.scatter(new_p[inside_local,0], new_p[inside_local,1], new_p[inside_local,2], c='r', marker='x', s=30, alpha=0.8, label=f'iter {iteration} interactions')
                ax.set_title(f"Iteration {iteration}: interaction positions (inside)")
                ax.set_xlabel('x [cm]')
                ax.set_ylabel('y [cm]')
                ax.set_zlabel('z [cm]')
                ax.legend()
                plt.draw(); plt.pause(pause_time)
                if debug_save:
                    fig.canvas.draw()
                    w,h = fig.canvas.get_width_height()
                    try:
                        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))
                    except Exception:
                        arr = np.asarray(fig.canvas.buffer_rgba())
                        if arr.ndim == 3 and arr.shape[2] == 4:
                            img = arr[:, :, :3].copy()
                        else:
                            img = arr.copy()
                    frames.append(img.copy())
            
            active_idx = np.where(active)[0]
            escaped = ~inside
            if np.any(escaped):
                idx_escaped = active_idx[escaped]
                no_deposit = E_depositata[idx_escaped] == 0
                E_depositata[idx_escaped[no_deposit]] = np.nan
                active[idx_escaped] = False

            inside_idx = active_idx[inside]
            if len(inside_idx) == 0:
                break  # no active photons remain

            pe_mask = (tipo[inside] == "Fotoelettrico")
            compton_mask = ~pe_mask
            print(f"E media: {np.mean(E[active_idx])}")
            print(f"Compton: {compton_mask.sum()}")
            print(f"PE: {pe_mask.sum()}\n")

            idx_pe = inside_idx[pe_mask]
            E_depositata[idx_pe] += E[idx_pe]
            E[idx_pe] = 0
            active[idx_pe] = False  # done

            idx_c = inside_idx[compton_mask]
            if len(idx_c) > 0:

                # Aggiorna angoli di scattering (vectorized by unique energies)
                scatter_angle = campiona_kn_array(THETA_MESH, E[idx_c])
                d[idx_c] = direzione_scatter(d[idx_c], scatter_angle)

                # Aggiorna posizioni
                new_p_active = new_p[inside]      # shape = (#inside, 3)
                new_p_compton = new_p_active[compton_mask]   # shape = (#compton, 3)
                p[idx_c] = new_p_compton

                new_E = compton(E[idx_c], scatter_angle)
                E_depositata[idx_c] += (E[idx_c] - new_E)
                E[idx_c] = new_E        

                # After updating positions and energies for Compton scatters,
                # show active photon positions for the next iteration (only those still active).
                iteration += 1
                if debug_graph:
                    ax.cla()
                    # Keep axes static across frames
                    r = volume.raggio
                    ax.set_xlim(-r, r)
                    ax.set_ylim(-r, r)
                    ax.set_zlim(0, volume.lunghezza)
                    if np.any(active):
                        pts_act = p[active]
                        ax.scatter(pts_act[:,0], pts_act[:,1], pts_act[:,2], c='b', marker='o', s=10, alpha=0.6, label=f'iter {iteration} active')
                    ax.set_title(f"Iteration {iteration}: active photon positions (inside)")
                    ax.set_xlabel('x [cm]')
                    ax.set_ylabel('y [cm]')
                    ax.set_zlabel('z [cm]')
                    ax.legend()
                    plt.draw(); plt.pause(pause_time)
                    if debug_save:
                        fig.canvas.draw()
                        w,h = fig.canvas.get_width_height()
                        try:
                            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))
                        except Exception:
                            arr = np.asarray(fig.canvas.buffer_rgba())
                            if arr.ndim == 3 and arr.shape[2] == 4:
                                img = arr[:, :, :3].copy()
                            else:
                                img = arr.copy()
                        frames.append(img.copy())
                print("=====================\n")
            # End while loop
        # Save animation if requested
        if debug_save and frames is not None and len(frames) > 0:
            # frames list created above
            if save_gif_path is None:
                save_gif_path = os.path.join('Montecarlo', 'Simulazioni', 'scatter_debug.gif')
            try:
                import imageio
                imageio.mimsave(save_gif_path, frames, duration=pause_time)
                print(f"Saved animation to {save_gif_path}")
            except Exception as e:
                try:
                    from PIL import Image
                    pil_imgs = [Image.fromarray(f) for f in frames]
                    pil_imgs[0].save(save_gif_path, save_all=True, append_images=pil_imgs[1:], duration=int(pause_time*1000), loop=0)
                    print(f"Saved animation to {save_gif_path} (PIL)")
                except Exception as ex:
                    print("Could not save animation (imageio/PIL not available):", ex)

        # Open a simple slider GUI to browse frames if requested
        if debug_slider and frames is not None and len(frames) > 0:
            try:
                from matplotlib.widgets import Slider
                ffig, fax = plt.subplots(figsize=(8,6))
                plt.subplots_adjust(bottom=0.2)
                imgplot = fax.imshow(frames[0])
                fax.set_axis_off()
                axslider = plt.axes([0.25, 0.05, 0.5, 0.03])
                slider = Slider(axslider, 'frame', 0, len(frames)-1, valinit=0, valfmt='%d')
                def update(val):
                    idx = int(slider.val)
                    imgplot.set_data(frames[idx])
                    ffig.canvas.draw_idle()
                slider.on_changed(update)
                plt.show()
            except Exception as e:
                print('Could not display slider for frames:', e)

        return E_depositata

    def scatter_through_volume(self, volume, debug_graph=False, pause_time=0.6, max_iters=1000):
        """Simulate Compton (and photoelectric) interactions inside a volume

        This function updates photon energies and directions as they undergo
        interactions inside `volume` (uses volume.cml). It returns the energy
        deposited inside the volume for each input photon and the subset of
        photons that exit the *front face* (z == volume.lunghezza). For those
        that exit through the front face we return their intersection position
        on the face and their outgoing directions.

        Returns:
            E_depositata: ndarray shape (N,) energy deposited inside volume
            exits: dict with keys 'idx', 'x','y','z','phi','psi','E' for photons
                   that exited through the front face. 'idx' are indices into
                   the original photon arrays.
        """
        N = len(self.energia)
        E = self.energia.copy()
        E_depositata = np.zeros_like(E)

        # positions and directions in local volume coordinates
        c = volume.centro_sup_vicina
        a = volume.angolo
        x,y,z = self.px-c[0], self.py-c[1], self.pz-c[2]
        # rotate coordinates by -a (same as scatter_inside)
        y,z = ((y*np.cos(a))-(z*np.sin(a))), ((y*np.sin(a))+(z*np.cos(a)))
        p = np.stack((x,y,z), axis=-1)

        phi, psi = self.phi - a, self.psi
        dx, dy, dz = np.sin(psi), np.sin(phi)*np.cos(psi), np.cos(phi)*np.cos(psi)
        d = np.stack((dx, dy, dz), axis=-1)

        active = np.ones(N, dtype=bool)
        exited_front_idx = []
        exited_front_pos = []
        exited_front_dir = []
        exited_front_energy = []

        it = 0
        while np.any(active) and it < max_iters:
            it += 1
            L, tipo = volume.cml(E[active])
            new_p = p[active] + d[active] * L[:, None]
            inside = self.isinside(new_p[:,0], new_p[:,1], new_p[:,2], volume)

            active_idx = np.where(active)[0]

            # For photons that would escape, check if they exit through the front face (z >= lunghezza)
            escaped = ~inside
            if np.any(escaped):
                esc_idx_local = np.where(escaped)[0]
                for j in esc_idx_local:
                    global_idx = active_idx[j]
                    p0 = p[global_idx]
                    d0 = d[global_idx]
                    L_j = L[j]
                    # If direction has positive z component, compute intersection with plane z = lunghezza
                    if d0[2] > 1e-12:
                        t_plane = (volume.lunghezza - p0[2]) / d0[2]
                        if t_plane >= 0 and t_plane <= L_j:
                            # intersection point
                            pt = p0 + d0 * t_plane
                            # check radial within radius
                            if (pt[0]**2 + pt[1]**2) <= (volume.raggio**2):
                                # record exit at front face
                                exited_front_idx.append(global_idx)
                                # transform back to global coordinates (undo local translation/rotation)
                                # inverse rotation by +a
                                y_pt = (pt[1]*np.cos(a) + pt[2]*np.sin(a))
                                z_pt = (-pt[1]*np.sin(a) + pt[2]*np.cos(a))
                                x_pt = pt[0] + c[0]
                                y_pt = y_pt + c[1]
                                z_pt = z_pt + c[2]
                                exited_front_pos.append((x_pt, y_pt, z_pt))
                                # compute outgoing angles phi,psi from d0
                                vec = d0 / np.linalg.norm(d0)
                                psi_out = np.arcsin(vec[0])
                                # handle phi via projection
                                phi_out = np.arctan2(vec[1], vec[2])
                                exited_front_dir.append((np.degrees(phi_out), np.degrees(psi_out)))
                                exited_front_energy.append(E[global_idx])
                                # mark photon as inactive (it left the volume)
                                active[global_idx] = False
                                continue
                    # otherwise it escaped laterally or backward: mark deposit NaN and deactivate
                    no_deposit = E_depositata[global_idx] == 0
                    if no_deposit:
                        E_depositata[global_idx] = np.nan
                    active[global_idx] = False

            # Now process those that are still inside after travelling L
            inside_idx_local = np.where(inside)[0]
            if len(inside_idx_local) == 0:
                break

            idx_inside_global = active_idx[inside_idx_local]
            pe_mask = (tipo[inside] == "Fotoelettrico")
            compton_mask = ~pe_mask

            # Photoelectric: deposit all energy and stop
            idx_pe = idx_inside_global[pe_mask]
            if len(idx_pe) > 0:
                E_depositata[idx_pe] += E[idx_pe]
                E[idx_pe] = 0
                active[idx_pe] = False

            # Compton: update directions and energies, move to interaction point
            idx_c = idx_inside_global[compton_mask]
            if len(idx_c) > 0:
                # compute scattering angles for compton interactions
                scatter_angle = campiona_kn_array(THETA_MESH, E[idx_c])
                # update directions
                d[idx_c] = direzione_scatter(d[idx_c], scatter_angle)
                # compute new positions at interaction points
                # need new_p active mapping
                new_p_active = new_p[inside]
                new_p_compton = new_p_active[compton_mask]
                p[idx_c] = new_p_compton
                # compute new energies after compton
                new_E = compton(E[idx_c], scatter_angle)
                E_depositata[idx_c] += (E[idx_c] - new_E)
                E[idx_c] = new_E

        # Prepare exit dict
        exits = {
            'idx': np.array(exited_front_idx, dtype=int) if len(exited_front_idx)>0 else np.array([], dtype=int),
            'x': np.array([p[0] for p in exited_front_pos]) if len(exited_front_pos)>0 else np.array([]),
            'y': np.array([p[1] for p in exited_front_pos]) if len(exited_front_pos)>0 else np.array([]),
            'z': np.array([p[2] for p in exited_front_pos]) if len(exited_front_pos)>0 else np.array([]),
            'phi': np.array([d[0] for d in exited_front_dir]) if len(exited_front_dir)>0 else np.array([]),
            'psi': np.array([d[1] for d in exited_front_dir]) if len(exited_front_dir)>0 else np.array([]),
            'E': np.array(exited_front_energy) if len(exited_front_energy)>0 else np.array([]),
        }

        return E_depositata, exits
        
######## Funzioni ########

def kn(theta, E):
    """"" Klein Nishima formula

    Parametri:
    theta: angolo di scattering
    E: energia del fotone incidente (keV) passato come nparray

    Retruns:
    Sezione d'urto differenziale
    """""
    E = float(E)
    epsilon = E/ME
    lr=1/(1+(epsilon*(1-np.cos(theta))))

    return 0.5*(RE**2)*(lr)**2*(lr + (lr)**-1 - (np.sin(theta))**2)

def campiona_kn(theta_mesh, E, N):
    """"" Campionamento della K-N usando tecniche numeriche

    Parametri:
    theta_mesh: Un mesh di theta da esplorare (linspace theta min-max)
    E: Energia del fotone incidente nparray
    N: Numero di campionamenti

    Returns: Numpy array di N angoli (in radianti) distribuiti secondo la KN normalizzata
    """""
    inv_cdf = _get_inv_cdf(theta_mesh, E)
    u = np.random.uniform(0,1, N)
    x = inv_cdf(u)
    # If a single sample was requested, return a scalar (not an array).
    if N == 1:
        return float(x)
    return x

# Cache for inv_cdf splines per energy (float key)
_campiona_kn_cache = {}

def _get_inv_cdf(theta_mesh, E):
    """Return a CubicSpline inv_cdf for given energy E, using a module-level cache."""
    # Use a simple float key; convert to float for consistency
    key = float(E)
    if key in _campiona_kn_cache:
        return _campiona_kn_cache[key]
    kn_norm = kn(theta_mesh, key) / integrate.quad(kn, 0, np.pi, args=(key))[0]
    cdf = np.cumsum(kn_norm) * (theta_mesh[1] - theta_mesh[0])
    cdf = cdf / cdf[-1]
    inv_cdf = interp.CubicSpline(cdf, theta_mesh)
    _campiona_kn_cache[key] = inv_cdf
    return inv_cdf

def campiona_kn_array(theta_mesh, energies):
    """Vectorized sampling of KN distribution for an array of energies.

    Parameters
    ----------
    theta_mesh : array-like
        Mesh of theta values used by the KN distribution.
    energies : array-like
        1D array of energies (one per sample) to draw from.

    Returns
    -------
    numpy.ndarray
        Array of sampled theta values, same shape as `energies`.
    """
    energies = np.asarray(energies)
    # get unique energies and counts
    unique, inv_idx, counts = np.unique(energies, return_inverse=True, return_counts=True)
    samples = np.empty_like(energies, dtype=float)
    # For each unique energy (identified by index j), sample count values using cached inv_cdf
    for j, u in enumerate(unique):
        pos = np.where(inv_idx == j)[0]
        cnt = len(pos)
        if cnt == 0:
            continue
        inv = _get_inv_cdf(theta_mesh, u)
        draws = inv(np.random.uniform(0, 1, cnt))
        samples[pos] = draws
    return samples

def direzione_scatter(d, theta):
    """ Calcola la nuova direzione della particella dopo uno scattering
    Parametri:
    d: (N,3) direzioni originali (xyz)
    theta: (N,) angolo di scattering

    Returns: (N,3) new direction vectors
    """
    d = d / np.linalg.norm(d, axis=1)[:,None]   # ensure normalized

    # Scegli vettore arbitrario, se s è già troppo vicino ad x usa z
    x = np.array([1., 0., 0.])
    mask = np.abs(d[:,0]) > 0.99
    a = np.where(mask[:,None], np.array([0.,0.,1.]), x)

    u = np.cross(a, d)
    u /= np.linalg.norm(u, axis=1)[:,None]
    
    v = np.cross(d, u)

    delta = np.random.uniform(-np.pi, np.pi, len(theta))
    new = (u * (np.sin(theta) * np.cos(delta))[:,None] +
           v * (np.sin(theta) * np.sin(delta))[:,None] +
           d * np.cos(theta)[:,None])

    return new

def compton(E, theta):
    """"" Calcola l'energia di un fotone entrante con energia E ed angolo theta

    Parametri:
    E: Energia in ingresso del fotone [keV]
    theta: angolo in ingresso (in radianti) del fotone

    Returns:
    Energia del fotone dopo lo scattering
    """""
    return E/(1+(E*(1-np.cos(theta))/ME))

def mc(E, phi_cristallo=PHI):
    sorgente    = Superficie(RCOL,(0,0,-DSP-L),0)
    collimatore = Superficie(RCOL,(0,0,-DSP), 0)
    plastico    = Superficie(RP, (0,0,0), 0)
    cristallo   = Superficie(RC,(0,DBC*np.sin(np.radians(phi_cristallo)), DBC*np.cos(np.radians(phi_cristallo))), phi_cristallo)
    C           = Materiale("C", 2.00)
    PMT1        = Volume(C, RP, LP, (0,0,0), 0)
    NaI         = Materiale("NaI", 3.67)
    PMT2        = Volume(NaI, RC, LC, (0,DBC*np.sin(np.radians(phi_cristallo)), DBC*np.cos(np.radians(phi_cristallo))), phi_cristallo)

    E = np.full(N_MC, E)
    ## Sorgente - collimatore
    xs, ys, zs = sorgente.pos_sul_piano_unif(N_MC, debug_graph=False) # Genera N punti uniformi sulla sorgente
    phiphi, psipsi = np.random.uniform(-SAACOLL, SAACOLL, len(xs)), np.random.uniform(-SAACOLL, SAACOLL, len(xs)) # Genera angoli uniformi
    f = Fotone(E, xs, ys, zs, phiphi, psipsi) # Genera fotoni 
    xc, yc, zc, phis, psis, E = f.calcola_int(collimatore, debug_graph=False) # Trova intersezione con collimatore

    # Collimatore - plastico
    f = Fotone(E, xc, yc, zc, phis, psis) # Fotoni sul collimatore con l'angolo da prima
    xp, yp, zp, phip, psip, E = f.calcola_int(plastico, debug_graph=False) # Trova intersezione con plastico

    f = Fotone(E, xp, yp, zp, phip, psip)
    # Simulate interactions inside the plastic volume and get exit positions
    _, exits = f.scatter_through_volume(PMT1, debug_graph=False)

    # Initialize other return values
    energie = np.array([])

    # Build arrays for photons that exited through front face
    if exits['idx'].size > 0:
        # The exits dict contains positions in global coordinates and outgoing angles (deg)
        x_exit = exits['x']
        y_exit = exits['y']
        z_exit = exits['z']
        phi_exit = exits['phi']
        psi_exit = exits['psi']
        E_exit = exits['E']
        # Create a Fotone for exited photons and intersect with the crystal
        f_exit = Fotone(E_exit, x_exit, y_exit, z_exit, phi_exit, psi_exit)
        xcr, ycr, zcr, phicr, psicr, energie = f_exit.calcola_int(cristallo, debug_graph=False)
        
        # If calcola_int returned all Nones (no hits on crystal), energie will be None
        if energie is not None:
            # Deposito d'energia dentro il cristallo
            f = Fotone(energie, xcr, ycr, zcr, phicr, psicr)
            energie = f.scatter_inside(PMT2, debug_graph=False, debug_save=False, save_gif_path=f'my_scatter_run{E}.gif', debug_slider=False)
        else:
            energie = np.array([])
    else:
        energie = np.array([])

    return energie

def plot_compton(phi_cristallo=PHI, all_peaks=False):
    energie1 = mc(E1 ,phi_cristallo)
    energie2 = mc(E2, phi_cristallo)
    energie1 = energie1[energie1 > ESOGLIA_C]
    energie2 = energie2[energie2 > ESOGLIA_C]
    

    sommato=np.concatenate((energie1, energie2)) if (len(energie1) > 0 and len(energie2) > 0) else np.array([])
    binss = np.linspace(ESOGLIA_C, EBINMAX, NBINS)

    plt.figure(figsize=(12,7), dpi=100)
    if all_peaks:
        if len(energie1) > 0:
            plt.hist(energie1, bins=binss, color="red", histtype="step", label=f"Picco del fotone di {round(E1,1)}keV", density=False)
        if len(energie2) > 0:
            plt.hist(energie2, bins=binss, color="blue", histtype="step", label=f"Picco del fotone di {round(E2,1)}keV", density=False)
    if len(sommato) > 0:
        plt.hist(sommato, bins=binss, color="black", histtype="step", label=f"Somma", density=False)
    

    plt.title(f"Segnale simulato per il cristallo posto a {phi_cristallo} gradi")
    plt.legend(loc="upper right")

    if len(sommato) > 0:
        en1 = np.pad(energie1, (0, len(sommato) - len(energie1)), constant_values=np.nan)
        en2 = np.pad(energie2, (0, len(sommato) - len(energie2)), constant_values=np.nan)
        
        # Create output directories if they don't exist
        os.makedirs(r'Montecarlo\Simulazioni\Istogrammi', exist_ok=True)
        os.makedirs(r'Montecarlo\Simulazioni\CSV', exist_ok=True)
        os.makedirs(r'Montecarlo\Simulazioni\Distribuzioni', exist_ok=True)
        
        file_path = os.path.join("Montecarlo", "Simulazioni", "Istogrammi", f"simul_picchi_{phi_cristallo}gradi.png")
        plt.savefig(file_path)

        file_path = os.path.join("Montecarlo", "Simulazioni", "CSV", f'simul_dati_{phi_cristallo}gradi.csv')
        np.savetxt(file_path, np.column_stack((en1, en2, sommato)), delimiter=',', header="Picco 1, Picco2, Segnale combinato")
    return sommato

######## Monte-Carlo ########
if __name__ == '__main__':
    start = time.time()
    plot_compton(phi_cristallo=35, all_peaks=True)
    end = time.time()
    print(f'Tempo impiegato: {round(end - start,2)}s')
    plt.show()

