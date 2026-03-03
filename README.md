# 🚗 Dashcam Computer Vision Lab

**Progetto di laboratorio — Computer Vision con OpenCV e Python**

---

## Descrizione del progetto

In questo laboratorio lavorerete in **gruppo** per costruire un sistema che analizza un video registrato da una dashcam (telecamera da auto) e **rileva oggetti** in tempo reale: pedoni, corsie stradali, semafori e altro.

Ogni gruppo si occupa di un **detector** diverso.  
Il programma principale (`main.py`) legge il video frame per frame, chiama tutti i detector attivi e disegna i risultati sullo schermo.

> **Non dovete costruire tutto da zero!**  
> Dovete solo scrivere il codice del vostro detector partendo da un file template già pronto.

---

## Obiettivo didattico

- Imparare le basi di **OpenCV** (lettura immagini, filtri, trasformazioni).
- Capire come funziona un **pipeline di computer vision** (video → frame → elaborazione → risultato).
- Lavorare in **gruppo** su un progetto condiviso.
- Scrivere codice **pulito e robusto** che si integra con il lavoro degli altri.

---

## Struttura delle cartelle

```
FlippedClassroom-ComputerVision/
│
├── main.py                          ← Programma principale (NON modificare)
│
├── assets/
│   └── dashcam.mp4                  ← Video dashcam di esempio
│
└── detectors/
    ├── base.py                      ← Classe base Detection (NON modificare)
    ├── template_group.py            ← Template da copiare per il vostro gruppo
    ├── pedestrians_hog.py           ← Esempio: rilevamento pedoni (HOG)
    ├── lanes_hough.py               ← Esempio: rilevamento corsie (Hough)
    └── trafficlight_color.py        ← Esempio: rilevamento semafori (colore HSV)
```

### File importanti

| File                              | Cosa fa                                | Si può modificare?       |
| --------------------------------- | -------------------------------------- | ------------------------ |
| `main.py`                         | Legge il video e chiama i detector     | **NO**                   |
| `detectors/base.py`               | Definisce `Detection` e `BaseDetector` | **NO**                   |
| `detectors/template_group.py`     | Template vuoto da copiare              | **NO** (copiatelo)       |
| `detectors/pedestrians_hog.py`    | Esempio di detector per pedoni         | Leggete come riferimento |
| `detectors/lanes_hough.py`        | Esempio di detector per corsie         | Leggete come riferimento |
| `detectors/trafficlight_color.py` | Esempio di detector per semafori       | Leggete come riferimento |

---

## Come installare le dipendenze

Aprite il terminale e digitate:

```bash
pip install opencv-python
```

Questo installa la libreria OpenCV per Python.

> **Nota:** se usate un ambiente virtuale (venv), attivatelo prima di eseguire il comando.

---

## Come avviare il progetto

```bash
python main.py
```

Si aprirà una finestra con il video della dashcam.  
Premete **`q`** per uscire.

> **Attenzione:** il file video deve trovarsi nella cartella `assets/dashcam.mp4`.
> Se non c'è, il programma darà errore.

---

## Cosa deve fare ogni gruppo (passo per passo)

### Passo 1 — Copiare il template

Copiate il file `detectors/template_group.py` e rinominatelo con il nome del vostro gruppo.

```bash
cp detectors/template_group.py detectors/group_pedoni.py
```

> Esempio di nomi: `group_pedoni.py`, `group_segnali.py`, `group_veicoli.py`

### Passo 2 — Cambiare il nome del detector

Aprite il vostro file e cambiate il nome:

```python
class Detector(BaseDetector):
    name = "group_pedoni"   # ← Mettete il nome del vostro gruppo
```

### Passo 3 — Scrivere il metodo `detect`

Dentro il metodo `detect`, scrivete il codice che:

1. Riceve un **frame** (immagine del video).
2. Elabora l'immagine con OpenCV (filtri, soglie, contorni...).
3. Trova gli oggetti.
4. Crea una lista di oggetti `Detection`.
5. Restituisce la lista.

### Passo 4 — Registrare il detector in `main.py`

Chiedete al docente di aggiungere il vostro detector in `main.py`:

```python
from detectors.group_pedoni import Detector as GruppoPedoniDetector

detectors = [
    PedDetector(),
    LaneDetector(),
    TLDetector(),
    GruppoPedoniDetector(),   # ← il vostro detector
]
```

### Passo 5 — Testare

Avviate il programma e controllate che il vostro detector funzioni:

```bash
python main.py
```

Se tutto va bene, vedrete i **rettangoli verdi** attorno agli oggetti rilevati.

---

## Formato obbligatorio della classe Detector

Il vostro file **deve** avere questa struttura:

```python
import cv2
import numpy as np
from detectors.base import BaseDetector, Detection

class Detector(BaseDetector):
    name = "nome_del_vostro_gruppo"

    def __init__(self):
        # Qui inizializzate variabili e parametri
        pass

    def detect(self, frame, frame_idx: int):
        h, w = frame.shape[:2]
        dets = []

        # --- Il vostro codice va qui ---

        return dets
```

### Regole del metodo `detect`

| Regola                           | Spiegazione                                   |
| -------------------------------- | --------------------------------------------- |
| Riceve `frame`                   | È un'immagine NumPy (BGR) del video           |
| Riceve `frame_idx`               | È il numero del frame corrente (0, 1, 2, ...) |
| Restituisce una **lista**        | Lista di oggetti `Detection`                  |
| Se non trova nulla → `return []` | **Mai** restituire `None`                     |
| Non deve **mai** generare errori | Usate `try/except` se necessario              |

### Cos'è un oggetto `Detection`

```python
Detection(x1, y1, x2, y2, label, conf)
```

| Campo   | Tipo  | Significato                                     |
| ------- | ----- | ----------------------------------------------- |
| `x1`    | int   | Coordinata X dell'angolo in alto a sinistra     |
| `y1`    | int   | Coordinata Y dell'angolo in alto a sinistra     |
| `x2`    | int   | Coordinata X dell'angolo in basso a destra      |
| `y2`    | int   | Coordinata Y dell'angolo in basso a destra      |
| `label` | str   | Nome dell'oggetto (es. `"pedestrian"`, `"car"`) |
| `conf`  | float | Confidenza da 0.0 a 1.0 (default: 1.0)          |

> **Importante:** chiamate sempre `.clamp(w, h)` per evitare coordinate fuori dall'immagine.

```python
dets.append(Detection(x1, y1, x2, y2, "car", 0.8).clamp(w, h))
```

---

## Esempio minimo di detector funzionante

Ecco un detector semplicissimo che rileva le zone **molto chiare** nell'immagine (potenziali luci):

```python
import cv2
import numpy as np
from detectors.base import BaseDetector, Detection

class Detector(BaseDetector):
    name = "group_luci"

    def __init__(self):
        self.soglia = 240  # soglia di luminosità

    def detect(self, frame, frame_idx: int):
        h, w = frame.shape[:2]
        dets = []

        # 1. Converti in scala di grigi
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. Applica una soglia: prendi solo i pixel molto luminosi
        _, thresh = cv2.threshold(gray, self.soglia, 255, cv2.THRESH_BINARY)

        # 3. Trova i contorni delle zone luminose
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        # 4. Per ogni contorno abbastanza grande, crea una Detection
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:        # ignora zone troppo piccole
                continue
            x, y, rw, rh = cv2.boundingRect(cnt)
            det = Detection(x, y, x + rw, y + rh, "bright_spot", 0.7)
            dets.append(det.clamp(w, h))

        return dets
```

### Cosa fa questo codice, riga per riga:

1. Converte il frame a colori in **scala di grigi**.
2. Applica una **soglia**: i pixel con valore > 240 diventano bianchi, gli altri neri.
3. Trova i **contorni** (bordi) delle zone bianche.
4. Per ogni contorno con area > 500 pixel, crea un rettangolo `Detection`.

---

## Regole importanti

- **Non modificare `main.py`** — il file è condiviso e deve funzionare per tutti.
- **Non modificare `base.py`** — contiene le classi base del progetto.
- **Non rinominare la classe** — deve chiamarsi sempre `Detector`.
- **Non cambiare la firma del metodo** — `detect(self, frame, frame_idx: int)` deve rimanere così.
- **Restituire sempre una lista** — anche vuota `[]`, mai `None`.
- **Non usare `cv2.imshow` nel detector** — la finestra la gestisce `main.py`.
- **Non usare `print` in eccesso** — rallenta il programma.
- **Testate spesso** — avviate `python main.py` dopo ogni modifica.

---

## Criteri di valutazione

| Criterio                                                    | Punti  |
| ----------------------------------------------------------- | ------ |
| Il detector **si avvia senza errori**                       | ⭐⭐   |
| Il detector **rileva qualcosa di sensato** nel video        | ⭐⭐⭐ |
| Il codice è **leggibile e commentato**                      | ⭐⭐   |
| Il gruppo sa **spiegare** come funziona il codice           | ⭐⭐   |
| Il detector ha **parametri regolabili** (soglie, ROI, ecc.) | ⭐     |

### In breve:

- **Sufficiente**: il detector parte e restituisce almeno qualche Detection.
- **Buono**: il detector rileva oggetti in modo sensato e il codice è commentato.
- **Ottimo**: il detector è preciso, ben strutturato e il gruppo sa spiegare ogni scelta.

---

## Problemi comuni e come risolverli

### Il programma non parte

```
RuntimeError: Impossibile aprire assets/dashcam.mp4
```

**Soluzione:** assicuratevi che il file `dashcam.mp4` sia nella cartella `assets/`.

---

### Errore di importazione

```
ModuleNotFoundError: No module named 'cv2'
```

**Soluzione:** installate OpenCV:

```bash
pip install opencv-python
```

---

### Il detector non viene chiamato

Controllate che sia stato aggiunto nella lista `detectors` in `main.py`:

```python
from detectors.group_pedoni import Detector as GruppoPedoniDetector

detectors = [
    ...
    GruppoPedoniDetector(),
]
```

---

### Il detector crasha e appare `[WARN]`

```
[WARN] detector group_pedoni crashed: ...
```

**Soluzione:** c'è un errore nel vostro codice `detect`. Leggete il messaggio di errore e correggete. Se non capite, provate a:

1. Aggiungere `print(frame.shape)` all'inizio di `detect` per controllare le dimensioni.
2. Avvolgere il codice in un blocco `try/except`:

```python
def detect(self, frame, frame_idx: int):
    h, w = frame.shape[:2]
    dets = []
    try:
        # il vostro codice qui
        pass
    except Exception as e:
        print(f"Errore nel detector: {e}")
    return dets
```

---

### I rettangoli non appaiono

- Controllate che `detect` restituisca una **lista di `Detection`**, non una lista di tuple.
- Controllate che le coordinate `(x1, y1, x2, y2)` siano numeri **interi** (`int`).
- Stampate il risultato per debug: `print(dets)`.

---

### Il video va molto lento

- Ridimensionate il frame prima di elaborarlo:

```python
scale = 0.5
small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
# lavorate su "small", poi moltiplicate le coordinate per 1/scale
```

- Non usate operazioni troppo pesanti su ogni frame.
- Elaborate un frame ogni N (es. `if frame_idx % 3 != 0: return []`).

---

## Funzioni OpenCV utili

Ecco alcune funzioni che vi serviranno spesso:

| Funzione                                | Cosa fa                                  |
| --------------------------------------- | ---------------------------------------- |
| `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` | Converte a scala di grigi                |
| `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)`  | Converte nello spazio colore HSV         |
| `cv2.GaussianBlur(img, (5,5), 0)`       | Sfoca l'immagine (riduce il rumore)      |
| `cv2.Canny(img, 50, 150)`               | Rileva i bordi                           |
| `cv2.threshold(img, 127, 255, ...)`     | Applica una soglia binaria               |
| `cv2.inRange(hsv, lower, upper)`        | Crea una maschera per un range di colore |
| `cv2.findContours(mask, ...)`           | Trova i contorni in un'immagine binaria  |
| `cv2.boundingRect(contour)`             | Rettangolo che contiene un contorno      |
| `cv2.HoughLinesP(edges, ...)`           | Rileva linee rette                       |
| `cv2.resize(img, (w, h))`               | Ridimensiona l'immagine                  |

---

## Risorse per studiare

- [Documentazione OpenCV Python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
- [Tutorial OpenCV in italiano (GeeksforGeeks)](https://www.geeksforgeeks.org/opencv-python-tutorial/)
- Guardate i file di esempio nella cartella `detectors/` per capire come sono fatti

---

> **Buon lavoro e buona visione!** 👁️
