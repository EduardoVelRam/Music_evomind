"""Audio WAV/MP3 -> características X con forma (ventanas, características).

Instalar: python -m pip install numpy matplotlib sounddevice "librosa>=0.11,<1" "soundfile>=0.13"
Ejecutar: python analizador_audio.py
Editar ARCHIVO_AUDIO y los parámetros del bloque principal antes de ejecutar.
"""

from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


class AnalizadorAudio:
    """Extrae características alineadas temporalmente, sin remuestrear el audio.

    X tiene una fila por ventana y una columna por característica. Por defecto:
    RMS, ZCR, centroide, n_mfcc coeficientes, magnitud STFT y fase (seno/coseno).
    Se mezclan los canales mediante su promedio. El audio original se conserva.
    Las ventanas están centradas en n * hop_length; fuera del fragmento se
    utilizan ceros. Se conserva también la ventana del borde final cuando
    su centro coincide con el final del fragmento, para permitir la inversión.
    """

    def __init__(self, n_fft=2048, hop_length=512, n_mfcc=13, n_mels=128):
        for nombre, valor in dict(n_fft=n_fft, hop_length=hop_length,
                                  n_mfcc=n_mfcc, n_mels=n_mels).items():
            if isinstance(valor, bool) or not isinstance(valor, (int, np.integer)):
                raise ValueError(f"{nombre} debe ser un entero.")
        if n_fft < 4 or n_fft % 2:
            raise ValueError("n_fft debe ser par y al menos 4.")
        if not 1 <= hop_length <= n_fft // 2:
            raise ValueError("hop_length debe estar entre 1 y n_fft//2 (solapamiento >=50%).")
        if not 1 <= n_mfcc <= n_mels <= n_fft // 2 + 1:
            raise ValueError("Se requiere 1 <= n_mfcc <= n_mels <= n_fft//2+1.")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.audio_original = None
        self.audio = None
        self.fs = None
        self.inicio_muestra = 0
        self.numero_muestras = None
        self._invalidar()

    def _invalidar(self):
        self._audio_padded = None
        self._magnitud = None
        self._stft = None
        self.tiempos = None
        self.caracteristicas = {}
        self.nombres = []
        self.X = None
        self.audio_generado = None
        self.fs_audio_generado = None

    def _comprobar_audio(self):
        if self.audio is None:
            raise RuntimeError("Primero llama a leer_audio().")

    def leer_audio(self, ruta):
        """Lee WAV/MP3; devuelve (audio_mono, fs) y selecciona todo el archivo.

        MP3 requiere una versión de libsndfile con soporte para ese formato.
        No se normaliza la amplitud ni se cambia la frecuencia de muestreo.
        """
        ruta = Path(ruta)
        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el archivo: {ruta}")
        datos, fs = sf.read(ruta, dtype="float64", always_2d=True)
        if datos.shape[0] == 0 or not np.isfinite(datos).all():
            raise ValueError("El audio está vacío o contiene muestras no finitas.")
        self.audio_original = datos.mean(axis=1)
        self.audio = self.audio_original.copy()
        self.fs = int(fs)
        self.numero_muestras = len(self.audio)
        self.inicio_muestra = 0
        self._invalidar()
        return self.audio.copy(), self.fs

    def obtener_secuencia(self, inicio=0.0, fin=None):
        """Selecciona [inicio, fin) en segundos respecto al archivo original.

        Devuelve las muestras del fragmento y lo deja activo para el análisis.
        fin=None significa final del archivo. Redondeo al índice más cercano.
        obtener_secuencia() recupera el audio completo. Invalida X anterior.
        """
        self._comprobar_audio()
        duracion = len(self.audio_original) / self.fs
        fin = duracion if fin is None else float(fin)
        inicio = float(inicio)
        if not (np.isfinite(inicio) and np.isfinite(fin)
                and 0 <= inicio < fin <= duracion):
            raise ValueError(f"Se requiere 0 <= inicio < fin <= {duracion:.6f} s.")
        i = int(round(inicio * self.fs))
        j = int(round(fin * self.fs))
        if i >= j:
            raise ValueError("El intervalo debe contener al menos una muestra.")
        self.audio = self.audio_original[i:j].copy()
        self.numero_muestras = len(self.audio)
        self.inicio_muestra = i
        self._invalidar()
        return self.audio.copy()

    def _preparar(self):
        self._comprobar_audio()
        if self._audio_padded is None:
            mitad = self.n_fft // 2
            self._audio_padded = np.pad(self.audio, (mitad, mitad))
            self._T = 1 + len(self.audio) // self.hop_length
            self.tiempos = (self.inicio_muestra
                            + np.arange(self._T) * self.hop_length) / self.fs

    def _stft_compleja(self):
        self._preparar()
        if self._stft is None:
            self._stft = librosa.stft(
                self._audio_padded, n_fft=self.n_fft,
                hop_length=self.hop_length, window="hann", center=False
            )[:, :self._T]
        return self._stft

    def _espectro(self):
        if self._magnitud is None:
            self._magnitud = np.abs(self._stft_compleja())
        return self._magnitud

    def obtener_magnitud(self):
        """Magnitud STFT sin normalizar, forma (T, n_fft//2+1)."""
        return self._espectro().T.copy()

    def obtener_fases(self, representacion="radianes"):
        """Fase por ventana/frecuencia, calculada con np.angle de la STFT.

        radianes: arreglo (T,F), valores entre -pi y pi.
        sincos: arreglo (T,2F): primero coseno y después seno.
        No se descartan fases de componentes débiles, para conservar precisión.
        """
        fase = np.angle(self._stft_compleja()).T
        if representacion == "radianes":
            return fase.copy()
        if representacion == "sincos":
            return np.concatenate([np.cos(fase), np.sin(fase)], axis=1)
        raise ValueError("representacion debe ser 'radianes' o 'sincos'.")

    def obtener_rms(self):
        """RMS temporal por ventana, forma (T,)."""
        self._preparar()
        return librosa.feature.rms(
            y=self._audio_padded, frame_length=self.n_fft,
            hop_length=self.hop_length, center=False, dtype=np.float64
        )[0, :self._T]

    def obtener_zcr(self):
        """Tasa de cruces por cero por ventana, forma (T,)."""
        self._preparar()
        return librosa.feature.zero_crossing_rate(
            self._audio_padded, frame_length=self.n_fft,
            hop_length=self.hop_length, center=False
        )[0, :self._T]

    def obtener_centroide(self):
        """Centroide espectral en Hz por ventana, forma (T,)."""
        return librosa.feature.spectral_centroid(S=self._espectro(), sr=self.fs)[0]

    def obtener_mfcc(self):
        """Coeficientes MFCC por ventana, forma (T, n_mfcc)."""
        mel = librosa.feature.melspectrogram(
            S=self._espectro() ** 2, sr=self.fs, n_fft=self.n_fft,
            n_mels=self.n_mels, fmax=self.fs / 2
        )
        log_mel = librosa.power_to_db(mel, ref=1.0, top_db=80.0)
        return librosa.feature.mfcc(S=log_mel, n_mfcc=self.n_mfcc).T

    def obtener_ancho_banda(self):
        """Dispersión espectral en Hz, forma (T,)."""
        return librosa.feature.spectral_bandwidth(S=self._espectro(), sr=self.fs)[0]

    def obtener_rolloff(self, porcentaje=0.85):
        """Frecuencia bajo la que queda el porcentaje de potencia indicado."""
        if not 0 < porcentaje < 1:
            raise ValueError("porcentaje debe estar entre 0 y 1.")
        return librosa.feature.spectral_rolloff(
            S=self._espectro() ** 2, sr=self.fs, roll_percent=porcentaje
        )[0]

    def obtener_planitud(self):
        """Planitud del espectro de potencia, forma (T,)."""
        return librosa.feature.spectral_flatness(S=self._espectro(), power=2.0)[0]

    def obtener_flujo(self):
        """Flujo positivo entre espectros de magnitud normalizados por su suma.

        Norma L2 de los incrementos positivos. El primer valor es cero.
        """
        S = self._espectro()
        S = S / np.maximum(S.sum(axis=0, keepdims=True), 1e-12)
        diferencia = np.maximum(np.diff(S, axis=1), 0.0)
        return np.r_[0.0, np.linalg.norm(diferencia, axis=0)]

    def obtener_croma(self):
        """Energía cromática normalizada, forma (T, 12); orden C, C#, ..., B.

        Afinación fija (tuning=0). En silencio se devuelven ceros.
        """
        S2 = self._espectro() ** 2
        if not np.any(S2):
            return np.zeros((self._T, 12))
        return librosa.feature.chroma_stft(
            S=S2, sr=self.fs, n_fft=self.n_fft, tuning=0.0
        ).T

    def obtener_caracteristicas(self, ampliadas=False, incluir_croma=False,
                               incluir_fase=True, incluir_magnitud=True,
                               representacion_fase="sincos"):
        """Calcula bloques (T, d); después llama a concatenar_matriz().

        Base: RMS, ZCR, centroide, MFCC. ampliadas agrega ancho de banda,
        rolloff de potencia al 85 %, planitud y flujo. Croma agrega 12 columnas.
        Por defecto agrega magnitud y fase STFT completas para reconstrucción.
        representacion_fase: 'sincos' (2F columnas) o 'radianes' (F columnas).
        """
        bloques = {
            "RMS": self.obtener_rms()[:, None],
            "ZCR": self.obtener_zcr()[:, None],
            "Centroide_Hz": self.obtener_centroide()[:, None],
        }
        if ampliadas:
            bloques.update({
                "Ancho_banda_Hz": self.obtener_ancho_banda()[:, None],
                "Rolloff85_Hz": self.obtener_rolloff()[:, None],
                "Planitud": self.obtener_planitud()[:, None],
                "Flujo": self.obtener_flujo()[:, None],
            })
        bloques["MFCC"] = self.obtener_mfcc()
        if incluir_croma:
            bloques["Croma"] = self.obtener_croma()
        if incluir_magnitud:
            bloques["Magnitud"] = self.obtener_magnitud()
        if incluir_fase:
            if representacion_fase == "sincos":
                fase = self.obtener_fases("sincos")
                F = self.n_fft // 2 + 1
                bloques["Fase_cos"] = fase[:, :F]
                bloques["Fase_sin"] = fase[:, F:]
            else:
                bloques["Fase_rad"] = self.obtener_fases(representacion_fase)
        self.caracteristicas = bloques
        self.nombres = []
        self.X = None
        self.audio_generado = None
        self.fs_audio_generado = None
        return {nombre: valores.copy() for nombre, valores in bloques.items()}

    def concatenar_matriz(self):
        """Concatena columnas y conserva X original sin normalizar, forma (T,D)."""
        if not self.caracteristicas:
            raise RuntimeError("Primero llama a obtener_caracteristicas().")
        nombres = []
        notas = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        for clave in self.caracteristicas:
            if clave == "MFCC":
                nombres.extend(f"MFCC_{i}" for i in range(self.n_mfcc))
            elif clave == "Croma":
                nombres.extend(f"Croma_{nota}" for nota in notas)
            elif clave in ("Magnitud", "Fase_rad", "Fase_cos", "Fase_sin"):
                nombres.extend(f"{clave}_{k}" for k in range(self.n_fft // 2 + 1))
            else:
                nombres.append(clave)
        X = np.concatenate(list(self.caracteristicas.values()), axis=1)
        if X.shape != (len(self.tiempos), len(nombres)) or not np.isfinite(X).all():
            raise ValueError("Las características no están alineadas o no son finitas.")
        self.X = X
        self.nombres = nombres
        return X.copy()

    def normalizar_matriz(self):
        """Min-max por columna para visualizar; columnas constantes -> cero.

        Conserva X. Para aprendizaje automático, ajustar el escalado solo con
        los datos de entrenamiento, en lugar de normalizar cada audio aparte.
        """
        if self.X is None:
            raise RuntimeError("Primero llama a concatenar_matriz().")
        minimo = self.X.min(axis=0)
        rango = np.ptp(self.X, axis=0)
        return (self.X - minimo) / np.where(rango > 0, rango, 1.0)

    def mostrar_imagen(self, normalizar=True, ruta_guardado=None, mostrar=True):
        """Visualiza X.T por bloques; espectros con eje de frecuencia en Hz.

        Devuelve (fig, axes). La normalización por columna solo afecta la
        imagen. Agrupa las miles de columnas espectrales en paneles compactos.
        """
        if self.X is None:
            raise RuntimeError("Primero llama a concatenar_matriz().")
        matriz = self.normalizar_matriz() if normalizar else self.X
        paso = self.hop_length / self.fs
        grupos = {}
        prefijos = ("Magnitud_", "Fase_rad_", "Fase_cos_", "Fase_sin_")
        for i, nombre in enumerate(self.nombres):
            grupo = next((p[:-1] for p in prefijos if nombre.startswith(p)), "Descriptores")
            grupos.setdefault(grupo, []).append(i)
        alturas = [max(3.3, 0.18 * len(indices)) if grupo == "Descriptores" else 3.3
                   for grupo, indices in grupos.items()]
        fig, axes = plt.subplots(len(grupos), 1, sharex=True, squeeze=False,
                                 figsize=(13, sum(alturas)),
                                 gridspec_kw={"height_ratios": alturas})
        axes = axes[:, 0]
        for ax, (grupo, indices) in zip(axes, grupos.items()):
            imagen = ax.imshow(
                matriz[:, indices].T, origin="lower", aspect="auto",
                interpolation="nearest", cmap="viridis",
                extent=[self.tiempos[0] - paso / 2, self.tiempos[-1] + paso / 2,
                        -0.5, len(indices) - 0.5],
                vmin=0 if normalizar else None, vmax=1 if normalizar else None
            )
            if grupo == "Descriptores":
                ax.set_yticks(np.arange(len(indices)),
                              labels=[self.nombres[i] for i in indices], fontsize=8)
                ax.set_ylabel("Característica")
            else:
                ticks = np.linspace(0, len(indices) - 1, min(6, len(indices)))
                ax.set_yticks(ticks, labels=[f"{k * self.fs / self.n_fft:.0f}" for k in ticks])
                ax.set_ylabel("Frecuencia [Hz]")
            ax.set_title(grupo.replace("_", " "))
            fig.colorbar(imagen, ax=ax, label="Escala por columna [0, 1]"
                         if normalizar else "Valor original")
        axes[-1].set_xlabel("Tiempo en el archivo original [s] (centro de ventana)")
        fig.suptitle(f"Matriz X: {self.X.shape[0]} ventanas × {self.X.shape[1]} características")
        fig.tight_layout()
        if ruta_guardado is not None:
            fig.savefig(ruta_guardado, dpi=180, bbox_inches="tight")
        if mostrar:
            plt.show()
        return fig, axes

    def guardar_matriz(self, ruta="caracteristicas_audio.npz"):
        """Guarda X, nombres, tiempos absolutos y parámetros en formato NumPy."""
        if self.X is None:
            raise RuntimeError("Primero llama a concatenar_matriz().")
        np.savez_compressed(
            ruta, X=self.X, nombres=np.asarray(self.nombres), tiempos=self.tiempos,
            fs=self.fs, n_fft=self.n_fft, hop_length=self.hop_length,
            n_mfcc=self.n_mfcc, n_mels=self.n_mels,
            inicio_muestra=self.inicio_muestra, numero_muestras=self.numero_muestras,
            version_formato=2, ventana="hann", centrado=True
        )

    @classmethod
    def cargar_matriz(cls, ruta):
        """Carga un NPZ versión 2 sin necesitar el audio original.

        Uso: analizador = AnalizadorAudio.cargar_matriz('caracteristicas_audio.npz')
        Luego: audio = analizador.generar_audio()
        """
        with np.load(ruta, allow_pickle=False) as datos:
            if "version_formato" not in datos or int(datos["version_formato"]) != 2:
                raise ValueError("Vuelve a extraer X con esta versión de AnalizadorAudio.")
            if str(datos["ventana"]) != "hann" or not bool(datos["centrado"]):
                raise ValueError("Se requiere ventana Hann y STFT centrada.")
            obj = cls(n_fft=int(datos["n_fft"]), hop_length=int(datos["hop_length"]),
                      n_mfcc=int(datos["n_mfcc"]), n_mels=int(datos["n_mels"]))
            obj.X = np.asarray(datos["X"], dtype=np.float64).copy()
            obj.nombres = datos["nombres"].tolist()
            obj.tiempos = np.asarray(datos["tiempos"], dtype=np.float64).copy()
            obj.fs = int(datos["fs"])
            obj.numero_muestras = int(datos["numero_muestras"])
            obj.inicio_muestra = int(datos["inicio_muestra"])
        if (obj.X.ndim != 2 or obj.X.shape[1] != len(obj.nombres)
                or obj.numero_muestras < 1 or obj.fs < 1
                or obj.X.shape[0] != 1 + obj.numero_muestras // obj.hop_length
                or obj.tiempos.shape != (obj.X.shape[0],)
                or not np.isfinite(obj.X).all() or not np.isfinite(obj.tiempos).all()
                or len(set(obj.nombres)) != len(obj.nombres)):
            raise ValueError("El archivo NPZ tiene dimensiones o metadatos inconsistentes.")
        return obj





if __name__ == "__main__":
    # Parámetros editables; no utiliza argparse.
    ARCHIVO_AUDIO = "audio.wav"  # También puede ser audio.mp3.
    INICIO = 19.0
    FIN = 24.0 #None  # Por ejemplo, INICIO=5.0 y FIN=10.0 seleccionan cinco segundos.

    analizador = AnalizadorAudio(n_fft=2048, hop_length=512, n_mfcc=13)
    audio, fs = analizador.leer_audio(ARCHIVO_AUDIO)
    fragmento = analizador.obtener_secuencia(INICIO, FIN)
    analizador.obtener_caracteristicas(
        ampliadas=False, incluir_croma=False,
        incluir_fase=True, incluir_magnitud=True,
        representacion_fase="sincos"  # También admite "radianes".
    )
    X = analizador.concatenar_matriz()

    print(f"Frecuencia de muestreo: {fs} Hz")
    print(f"Duración del fragmento: {len(fragmento) / fs:.3f} s")
    print(f"Ventana: {1000 * analizador.n_fft / fs:.2f} ms")
    print(f"Salto: {1000 * analizador.hop_length / fs:.2f} ms")
    print(f"X: {X.shape} (ventanas, características)")
    print("Primeras columnas:", analizador.nombres[:8])
    analizador.guardar_matriz("caracteristicas_audio.npz")
    # Reconstrucción desde magnitud+fase guardadas dentro de X sin normalizar.
    generado = analizador.generar_audio(
        X=X, metodo="auto", ruta_guardado="audio_generado1.wav"
    )
    print(f"Audio generado: {len(generado)} muestras")
    print(f"Error máximo respecto al fragmento mono: {np.max(np.abs(fragmento - generado)):.3e}")
    analizador.reproducir_audio(volumen=0.5, esperar=True)
    analizador.mostrar_imagen(ruta_guardado="matriz_caracteristicas.png")