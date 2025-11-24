"""
Utilitats per a l'anàlisi astrosismològica.

Aquest mòdul conté les funcions principals per analitzar espectres de freqüències
estel·lars i detectar pics.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from typing import List, Tuple, Optional
from scipy.signal import find_peaks


def load_data(path: str, freq_min: Optional[float] = None, freq_unit: str = 'mHz') -> Tuple[np.ndarray, np.ndarray]:
    """Carrega les dades de freqüència i amplitud des d'un fitxer CSV.
    
    Format esperat: columnes 'Frequencia' i 'Amplitud'
    (utilitzar normalitzar_csvs.py per normalitzar noms de columnes)
    
    Paràmetres:
        path: Ruta al fitxer CSV
        freq_min: Freqüència mínima per filtrar soroll (en les mateixes unitats que les dades). 
                  Si és None, no filtra.
        freq_unit: Unitat de freqüència de les dades ('mHz' o 'microHz'). 
                   Només informatiu per documentació.
    
    Retorna:
        Tuple amb arrays de freqüències i amplituds (en les unitats originals del fitxer)
    """
    freqs: list[float] = []
    amps: list[float] = []
    dropped = 0
    
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Fitxer buit: {path}")

        header_norm = [h.strip() for h in header]
        
        # Busquem les columnes estàndard
        try:
            freq_idx = header_norm.index('Frequencia')
            amp_idx = header_norm.index('Amplitud')
        except ValueError:
            raise ValueError(f"Format de columnes no reconegut. S'esperen 'Frequencia' i 'Amplitud'. "
                           f"Trobat: {header_norm}\n"
                           f"Executar normalitzar_csvs.py per normalitzar els noms de columnes.")

        for row in reader:
            if len(row) <= max(freq_idx, amp_idx):
                dropped += 1
                continue

            fs = row[freq_idx].strip()
            as_ = row[amp_idx].strip()

            try:
                fval = float(fs)
                aval = float(as_)
                
                # Filtra freqüències baixes si s'especifica
                if freq_min is not None and fval < freq_min:
                    continue
                    
            except Exception:
                dropped += 1
                continue

            freqs.append(fval)
            amps.append(aval)

    if dropped:
        print(f'Avís: s\'han descartat {dropped} files de {path}')
    if len(freqs) < 3:
        raise ValueError('No hi ha prou mostres numèriques vàlides després d\'analitzar l\'entrada')

    return np.array(freqs, dtype=float), np.array(amps, dtype=float)


def select_peaks_by_amplitude(peaks: List[int], amps: np.ndarray, freqs: np.ndarray, 
                              count: int, freq_min: float = 0.0, freq_max: float = 10000.0) -> List[int]:
    """Selecciona els pics més alts per amplitud dins d'un rang de freqüència.
    
    Paràmetres:
        peaks: Llista d'índexs de pics
        amps: Array d'amplituds
        freqs: Array de freqüències
        count: Nombre de pics a seleccionar
        freq_min: Freqüència mínima a considerar
        freq_max: Freqüència màxima a considerar
    
    Retorna:
        Llista d'índexs de pics seleccionats ordenats per freqüència
    """
    # Filtrar pics per rang de freqüència
    filtered_peaks = [p for p in peaks if freq_min <= freqs[p] <= freq_max]
    
    # Ordenar per amplitud i seleccionar els més alts
    peaks_by_amp = sorted(filtered_peaks, key=lambda i: amps[i], reverse=True)
    selected = peaks_by_amp[:count]
    
    # Retornar ordenats per freqüència ascendent
    return sorted(selected)


def process_spectrum(freqs: np.ndarray, amps: np.ndarray,
                    num_peaks: int = 300, 
                    freq_range_min: float = 0.0, 
                    freq_range_max: float = 10000.0,
                    prominence: float = 0.5,
                    distance: int = 2,
                    freq_unit: str = 'microHz') -> Tuple[List[int], List[int], List[int], int]:
    """Processa l'espectre de freqüències detectant i seleccionant pics.
    
    Aquesta funció unificada serveix tant per estrelles com per el Sol:
    - Troba el màxim dins del rang especificat
    - Detecta pics amb scipy.signal.find_peaks
    - Selecciona els pics més alts per amplitud dins del rang
    
    Paràmetres:
        freqs: Array de freqüències
        amps: Array d'amplituds
        num_peaks: Nombre de pics a seleccionar
        freq_range_min: Freqüència mínima del rang de selecció
        freq_range_max: Freqüència màxima del rang de selecció
        prominence: Prominència mínima per detectar pics (en dB)
        distance: Distància mínima entre pics (en mostres)
        freq_unit: Unitat de freqüència (per missatges)
    
    Retorna:
        Tuple (initial_peaks, selected_peaks, global_max_idx)
    """
    # Màxim dins del rang
    mask = (freqs >= freq_range_min) & (freqs <= freq_range_max)
    indices_in_range = np.where(mask)[0]
    if len(indices_in_range) > 0:
        global_max_idx = indices_in_range[np.argmax(amps[indices_in_range])]
        print(f'Màxim en rang {freq_range_min}-{freq_range_max} {freq_unit}: {freqs[global_max_idx]:.6f} {freq_unit}')
    else:
        global_max_idx = int(np.argmax(amps))
        print(f'⚠ No hi ha dades en el rang especificat. Usant màxim global: {freqs[global_max_idx]:.6f} {freq_unit}')
    
    # Detectar pics amb scipy.signal.find_peaks
    peaks, _ = find_peaks(amps, prominence=prominence, distance=distance)
    initial_peaks = list(peaks)
    print(f'Pics detectats amb find_peaks (prominence={prominence} dB, distance={distance}): {len(initial_peaks)}')
    
    # Seleccionar pics per amplitud dins del rang
    selected_peaks = select_peaks_by_amplitude(
        initial_peaks, amps, freqs, count=num_peaks,
        freq_min=freq_range_min, freq_max=freq_range_max
    )
    print(f'Pics seleccionats per amplitud (rang {freq_range_min}-{freq_range_max} {freq_unit}): {len(selected_peaks)}')
    
    return initial_peaks, selected_peaks, global_max_idx


def compute_pairwise_differences(freqs: np.ndarray, peaks: List[int]):
    """Calcula totes les diferències de freqüència entre parells de pics.
    
    Paràmetres:
        freqs: Array de freqüències
        peaks: Llista d'índexs dels pics seleccionats
    
    Retorna:
        Llista de tuples (etiqueta_parell, diferència_mHz)
    """
    peak_freqs = np.sort(freqs[peaks])
    n = len(peak_freqs)
    pairs = []
    
    for i in range(n - 1):
        for j in range(i + 1, n):
            delta = float(peak_freqs[j] - peak_freqs[i])
            pair_label = f"{peak_freqs[j]:.6f}-{peak_freqs[i]:.6f}"
            pairs.append((pair_label, delta))
    
    return pairs


def plot_spectrum(freqs: np.ndarray, amps: np.ndarray, 
                  initial_peaks: List[int] = None,
                  selected_peaks: List[int] = None,
                  global_max_idx: int = None,
                  title: str = 'Espectre de freqüències',
                  freq_unit: str = 'mHz'):
    """Visualitza l'espectre amb pics marcats.
    
    Paràmetres:
        freqs: Array de freqüències
        amps: Array d'amplituds
        initial_peaks: Pics detectats (opcional)
        selected_peaks: Pics seleccionats finals (opcional)
        global_max_idx: Índex del màxim global (opcional)
        title: Títol de la gràfica
        freq_unit: Unitat de freqüència ('mHz' o 'microHz')
    """
    # Ordenem per freqüència
    order = np.argsort(freqs)
    freqs_sorted = freqs[order]
    amps_sorted = amps[order]
    
    plt.figure(figsize=(14, 6))
    plt.plot(freqs_sorted, amps_sorted, '-', color='black', linewidth=0.8, label='Espectre', alpha=0.6)
    
    # Marquem els diferents tipus de pics
    if initial_peaks is not None:
        plt.scatter(freqs[initial_peaks], amps[initial_peaks], color='lightgray', s=12, 
                    zorder=3, label=f'Pics detectats ({len(initial_peaks)})', alpha=0.5)
    if selected_peaks is not None:
        plt.scatter(freqs[selected_peaks], amps[selected_peaks], color='red', s=40, 
                    zorder=6, label=f'Pics seleccionats ({len(selected_peaks)})')
    if global_max_idx is not None:
        plt.scatter([freqs[global_max_idx]], [amps[global_max_idx]], color='blue', s=100, 
                    marker='*', zorder=7, label='Central (màxim global)')
    
    plt.xlabel(f'Freqüència ({freq_unit})')
    plt.ylabel('Amplitud (dB)')
    plt.title(title)
    plt.legend(loc='best', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def compute_autocorrelation(freqs: np.ndarray, amps: np.ndarray, 
                           freq_range_min: float = None, freq_range_max: float = None,
                           exclude_near_zero: float = 0.05):
    """Calcula l'autocorrelació de l'espectre freqüencial en el rang útil.
    
    L'autocorrelació es calcula sobre l'espectre de potència dins del rang
    de freqüències especificat per trobar periodicitats.
    
    Paràmetres:
        freqs: Array de freqüències completes
        amps: Array d'amplituds corresponents
        freq_range_min: Freqüència mínima del rang útil (opcional)
        freq_range_max: Freqüència màxima del rang útil (opcional)
        exclude_near_zero: Fracció de l'interval a excloure prop de zero (default 0.05 = 5%)
    
    Retorna:
        Tuple (lags, autocorr, peak_lag): lags en unitats de freqüència, 
                                          autocorrelació normalitzada, 
                                          i lag del pic màxim (espaiat mitjà entre pics)
    """
    # Ordenar per freqüències
    sorted_idx = np.argsort(freqs)
    freqs_sorted = freqs[sorted_idx]
    amps_sorted = amps[sorted_idx]
    
    # Filtrar pel rang útil si s'especifica
    if freq_range_min is not None or freq_range_max is not None:
        mask = np.ones(len(freqs_sorted), dtype=bool)
        if freq_range_min is not None:
            mask &= (freqs_sorted >= freq_range_min)
        if freq_range_max is not None:
            mask &= (freqs_sorted <= freq_range_max)
        freqs_sorted = freqs_sorted[mask]
        amps_sorted = amps_sorted[mask]
    
    # Interpolar l'espectre a un grid uniforme per poder fer l'autocorrelació
    freq_min = freqs_sorted.min()
    freq_max = freqs_sorted.max()
    
    # Usar la resolució del grid original
    freq_spacing = np.median(np.diff(freqs_sorted))
    grid = np.arange(freq_min, freq_max + freq_spacing, freq_spacing)
    
    # Interpolar amplituds al grid uniforme
    signal = np.interp(grid, freqs_sorted, amps_sorted)
    
    # Calcular autocorrelació
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr / autocorr.max()  # Normalitzar
    
    # Agafar només la part positiva (lags >= 0)
    mid = len(autocorr) // 2
    autocorr = autocorr[mid:]
    lags = np.arange(len(autocorr)) * freq_spacing
    
    # Trobar pics de l'autocorrelació amb scipy.signal.find_peaks
    # Definir rang de cerca: excloure prop de zero
    exclude_lag = lags.max() * exclude_near_zero
    max_lag_search = lags.max() * 0.5  # Cercar fins al 50% del rang
    
    # Filtrar lags vàlids
    valid_mask = (lags > exclude_lag) & (lags <= max_lag_search)
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) > 0:
        autocorr_range = autocorr[valid_indices]
        lags_range = lags[valid_indices]
        
        # Trobar pics amb prominència mínima
        peak_indices, properties = find_peaks(autocorr_range, 
                                              prominence=0.01,
                                              distance=3)
        
        if len(peak_indices) > 0:
            # Ordenar pics per prominència (descendent)
            prominences = properties['prominences']
            sorted_idx = np.argsort(prominences)[::-1]
            
            # Agafar el pic més prominent
            main_peak_idx = peak_indices[sorted_idx[0]]
            peak_lag = lags_range[main_peak_idx]
        else:
            # Si no es troben pics, agafar el màxim
            peak_lag = lags_range[np.argmax(autocorr_range)]
    else:
        # Fallback: agafar el màxim global
        peak_lag = lags[np.argmax(autocorr[1:])] if len(autocorr) > 1 else lags[0]
    
    return lags, autocorr, peak_lag


def plot_histogram_with_autocorrelation(deltas: np.ndarray, 
                                        lags: np.ndarray = None, 
                                        autocorr: np.ndarray = None,
                                        autocorr_peak: float = None, 
                                        bin_width: float = 0.03,
                                        freq_unit: str = 'microHz', 
                                        exclude_near_zero: bool = True):
    """Crea una visualització d'histograma amb autocorrelació opcional.
    
    Si es proporcionen lags i autocorr, mostra histograma + autocorrelació en 2 subplots.
    Si no, mostra només l'histograma.
    
    Paràmetres:
        deltas: Array de diferències de freqüència
        lags: Array de lags de l'autocorrelació (opcional)
        autocorr: Array de valors d'autocorrelació (opcional)
        autocorr_peak: Posició del pic màxim de l'autocorrelació (opcional)
        bin_width: Amplada del bin
        freq_unit: Unitat de freqüència ('mHz' o 'microHz')
        exclude_near_zero: Si True, no marca el bin més freqüent si està prop de zero
    """
    # Preparar histograma
    max_d = deltas.max()
    bins = np.arange(0.0, max_d + bin_width, bin_width)
    counts, edges = np.histogram(deltas, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    
    # Determinar si mostrem autocorrelació
    show_autocorr = lags is not None and autocorr is not None
    
    # Crear figura
    if show_autocorr:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
    
    # Subplot 1: Histograma
    ax1.bar(edges[:-1], counts, width=bin_width, align='edge', edgecolor='k', alpha=0.7)
    
    # Marcar bin més freqüent
    exclude_threshold = max_d * 0.05 if exclude_near_zero else 0.0
    max_count = int(counts.max())
    top_idxs = np.where(counts == max_count)[0]
    for idx in top_idxs:
        c = centers[idx]
        if c > exclude_threshold:
            ax1.axvline(c, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ypos = counts[idx] + max(1, int(0.02 * max_count))
            ax1.text(c, ypos, f"{c:.3f}\n{counts[idx]}", color='red', ha='center', 
                     va='bottom', fontsize=9, weight='bold')
    
    # Marcar pic d'autocorrelació si existeix
    if autocorr_peak is not None:
        label = f'Δν (autocorrelació) = {autocorr_peak:.3f} {freq_unit}' if show_autocorr else f'Pic autocorrelació: {autocorr_peak:.3f} {freq_unit}'
        ax1.axvline(autocorr_peak, color='green' if show_autocorr else 'blue', 
                   linestyle=':', linewidth=2.5, alpha=0.8, label=label)
    
    ax1.set_ylabel('Nombre de parells', fontsize=12, fontweight='bold')
    title = 'Histograma de diferències de freqüència'
    if not show_autocorr:
        title += f' (amplada bin = {bin_width} {freq_unit})'
    ax1.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    if autocorr_peak is not None or not show_autocorr:
        ax1.legend(loc='best', fontsize=10)
    
    if not show_autocorr:
        # Etiquetes eix X només si no hi ha autocorrelació
        step = max(1, len(centers) // 40)
        ax1.set_xticks(centers[::step])
        ax1.set_xticklabels([f"{c:.3f}" for c in centers[::step]], rotation=90, fontsize=8)
        ax1.set_xlim(edges[0], edges[-1])
        ax1.set_xlabel(f'Diferència de freqüència ({freq_unit})')
    
    # Subplot 2: Autocorrelació (només si es proporciona)
    if show_autocorr:
        ax2.plot(lags, autocorr, 'b-', linewidth=1.5, alpha=0.8)
        
        if autocorr_peak is not None:
            # Trobar el valor real en l'autocorrelació
            idx_delta_nu = np.argmin(np.abs(lags - autocorr_peak))
            actual_peak_value = autocorr[idx_delta_nu]
            
            ax2.axvline(autocorr_peak, color='green', linestyle='--', linewidth=2.5,
                       label=f'Δν = {autocorr_peak:.3f} {freq_unit}', zorder=5)
            ax2.plot(autocorr_peak, actual_peak_value, 'g*', markersize=20, zorder=6)
            
            # Harmònics (2x, 3x)
            max_lag = lags.max()
            for mult in [2, 3]:
                harmonic = autocorr_peak * mult
                if harmonic <= max_lag * 0.6:
                    ax2.axvline(harmonic, color='orange', linestyle=':', linewidth=1.5,
                               alpha=0.6, label=f'{mult}×Δν' if mult == 2 else None)
        
        ax2.set_xlabel(f'Δν ({freq_unit})', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Autocorrelació', fontsize=12, fontweight='bold')
        ax2.set_title('Autocorrelació de l\'espectre', fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax2.legend(loc='best', fontsize=10)
        
        # Sincronitzar eixos X
        x_max = min(max_d, lags.max() * 0.6)
        ax1.set_xlim(0, x_max)
        ax2.set_xlim(0, x_max)
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar estadístiques dels bins amb més parells
    print(f"\nBins amb més parells (excloent zona < {exclude_threshold:.3f} {freq_unit}):")
    valid_indices = centers > exclude_threshold
    valid_centers = centers[valid_indices]
    valid_counts = counts[valid_indices]
    
    top_bins = np.argsort(valid_counts)[::-1][:5]
    for idx in top_bins:
        if idx < len(valid_centers):
            print(f"  {valid_centers[idx]:.3f} {freq_unit}: {valid_counts[idx]} parells")


def save_results(freqs: np.ndarray, amps: np.ndarray, 
                 selected_peaks: List[int], global_max_idx: int,
                 pairwise_diffs: List[Tuple[str, float]],
                 counts: np.ndarray, edges: np.ndarray,
                 output_dir: str = 'output', bin_width: float = 0.03,
                 freq_unit: str = 'mHz'):
    """Guarda tots els resultats en fitxers CSV.
    
    Paràmetres:
        freqs: Array de freqüències
        amps: Array d'amplituds
        selected_peaks: Llista d'índexs dels pics seleccionats
        global_max_idx: Índex del màxim global
        pairwise_diffs: Llista de diferències entre parells
        counts: Comptadors de l'histograma
        edges: Vores dels bins de l'histograma
        output_dir: Directori de sortida
        bin_width: Amplada del bin
        freq_unit: Unitat de freqüència ('mHz' o 'microHz')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Pics seleccionats
    peaks_file = os.path.join(output_dir, 'peaks_around_central.csv')
    with open(peaks_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        # Utilitzar la unitat correcta segons freq_unit
        freq_col_name = f'Frequency ({freq_unit})'
        writer.writerow(['Index', freq_col_name, 'Amplitude (dB)', 'Distance from central (samples)'])
        for idx in selected_peaks:
            writer.writerow([
                int(idx),
                f"{float(freqs[idx]):.8f}",
                f"{float(amps[idx]):.8f}",
                int(abs(idx - global_max_idx))
            ])
    print(f"Pics seleccionats guardats a: {peaks_file}")
    
    # Diferències entre parells
    pairs_file = os.path.join(output_dir, 'pairwise_differences.csv')
    with open(pairs_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Pair', f'Delta ({freq_unit})'])
        for pair, delta in pairwise_diffs:
            writer.writerow([pair, f"{float(delta):.6f}"])
    print(f"Diferències entre parells guardades a: {pairs_file}")
    
    # Histograma
    hist_file = os.path.join(output_dir, f'histogram_{bin_width:.2f}{freq_unit}.csv')
    with open(hist_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([f'BinStart({freq_unit})', f'BinEnd({freq_unit})', f'BinCenter({freq_unit})', 'Count'])
        for c, left, right in zip(counts, edges[:-1], edges[1:]):
            writer.writerow([f"{float(left):.6f}", f"{float(right):.6f}", 
                            f"{float((left+right)/2):.6f}", int(c)])
    print(f"Histograma guardat a: {hist_file}")
