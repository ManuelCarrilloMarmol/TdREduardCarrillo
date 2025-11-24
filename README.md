# Astrosismologia (TdR Eduard Carrillo González)

Scripts de python per analitzar l'espectre frequencial de senyals d'origen sísmic de diverses estrelles i calcular diferents paràmetres de les mateixes com per exemple el seu radi i la seva massa.

## Estructura del projecte

```
TdR/
├── dades/                          # Fitxers CSV amb espectres
│   ├── sol.csv
│   ├── estrellaA.csv
│   ├── estrellaB.csv
│   ├── estrellasC.csv
│   └── estrellaD.csv
├── astrosismologia_utils.py       # Funcions reutilitzables
├── configuracio_estrelles.py      # Paràmetres per cada estrella
├── executar_parametritzat.py      # Script per processar totes les estrelles
├── analisi_simple.ipynb           # Notebook parametritzable
└── output/                         # Resultats generats
    ├── sol/
    ├── estrellaA/
    └── ...
```

## Dependències

Requereix els següents paquets:

- numpy (operacions amb arrays)
- matplotlib (gràfiques)
- jupyter (entorn interactiu per visualitzar i analitzar les dades)
- papermill (execució parametritzada de notebooks)

Estan llistats a `requeriments.txt` a l'arrel del repositori.

## Configuració ràpida (Windows / PowerShell)

1. Assegura't de tenir Python 3.8+ instal·lat i disponible al PATH.
2. Crea i activa un entorn virtual (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Instal·la les dependències (això "restaurarà" les dependències del projecte):

```powershell
# Si 'pip' no està disponible al PATH, utilitza la forma amb el mòdul de Python que
# funciona amb el mateix intèrpret que has invocat. Exemple:
python -m pip install -r requirements.txt

# Alternativament, pots utilitzar l'script inclòs en aquest repositori per crear
# el venv i instal·lar dependències sense dependre de 'pip' al PATH:
.\restore-deps.ps1
```

## Ús

### Opció 1: Processar totes les estrelles automàticament

```powershell
python executar_parametritzat.py
```

Això processarà el Sol i les estrelles A, B, C i D amb els paràmetres òptims per a cadascuna, 
generant notebooks i resultats CSV per separat.

### Opció 2: Notebook interactiu amb widgets

Obre `analisi.ipynb` a Jupyter i utilitza els controls interactius per ajustar 
paràmetres visualment.

### Opció 3: Notebook simple parametritzable

```bash
# Amb papermill
papermill analisi.ipynb output.ipynb -p DATA_FILE dades/estrellaA.csv -p FREQ_MIN 0.05

# O executa directament
jupyter notebook analisi.ipynb
```

## Configuració per estrella

Cada estrella té paràmetres optimitzats a `configuracio_estrelles.py`:

- **Sol**: Sense filtre de freqüència, bin de 0.03 mHz
- **Estrelles A-D**: Filtre > 0.05 mHz (elimina soroll), bin de 0.002 mHz


## Resultats

Cada execució genera:
- `peaks_around_central.csv` - Pics seleccionats al voltant del central
- `pairwise_differences.csv` - Diferències entre tots els parells de pics
- `histogram_X.XXmHz.csv` - Histograma de diferències