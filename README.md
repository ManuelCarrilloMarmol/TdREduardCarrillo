# Astrosismologia (TdR Eduard Carrillo González)

Scripts de python per analitzar l'espectre frequencial de senyals d'origen sísmic de diverses estrelles i calcular diferents paràmetres de les mateixes com per exemple el seu radi i la seva massa.

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