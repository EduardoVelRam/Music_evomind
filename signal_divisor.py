
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import butter, sosfiltfilt, sosfilt

# intensidad RMS (Root Mean Square o valor cuadrático medio)

def separar_componentes(
    nombre_audio,
    inicio=0,
    fin=None,
    bandas=None,
    carpeta_audio="audios",
    carpeta_salida="audios/raymix",
    orden=5,
    ventana_rms=0.05
):
    """
    Separa un intervalo de audio en diferentes bandas de frecuencia,
    guarda las señales resultantes y genera una gráfica de intensidad RMS.

    Parámetros
    ----------
    nombre_audio : str
        Nombre del archivo ubicado en la carpeta de audios.

    inicio : float
        Segundo inicial del intervalo.

    fin : float o None
        Segundo final. Si es None, utiliza el final del audio.

    bandas : dict
        Diccionario con las bandas de frecuencia.
        Ejemplo:
        {
            "graves": (20, 250),
            "medios": (250, 4000),
            "agudos": (4000, 10000)
        }

    carpeta_audio : str
        Directorio del audio original.

    carpeta_salida : str
        Directorio de los archivos generados.

    orden : int
        Orden del filtro Butterworth.

    ventana_rms : float
        Duración de la ventana RMS en segundos.

    Retorna
    -------
    dict
        Señales, rutas de archivos y datos de intensidad.
    """

    # --------------------------------------------------
    # Configuración de bandas
    # --------------------------------------------------

    if bandas is None:
        bandas = {
            "graves": (20, 250),
            "medios": (250, 4000),
            "agudos": (4000, 10000)
        }

    # --------------------------------------------------
    # Cargar audio
    # --------------------------------------------------

    ruta_audio = Path(carpeta_audio) / nombre_audio

    if not ruta_audio.is_file():
        raise FileNotFoundError(
            f"No se encontró el audio: {ruta_audio}"
        )

    audio, frecuencia_muestreo = sf.read(
        ruta_audio,
        dtype="float32",
        always_2d=True
    )

    duracion = len(audio) / frecuencia_muestreo

    # --------------------------------------------------
    # Validar intervalo temporal
    # --------------------------------------------------

    if fin is None:
        fin = duracion

    if inicio < 0 or fin <= inicio or fin > duracion:
        raise ValueError(
            f"Intervalo inválido. El audio dura {duracion:.2f} segundos."
        )

    muestra_inicio = int(inicio * frecuencia_muestreo)
    muestra_fin = int(fin * frecuencia_muestreo)

    # --------------------------------------------------
    # Preparar audio
    # --------------------------------------------------

    audio = audio.T
    nyquist = frecuencia_muestreo / 2

    directorio = Path(carpeta_salida)
    directorio.mkdir(parents=True, exist_ok=True)

    nombre_base = ruta_audio.stem

    componentes = {}

    # --------------------------------------------------
    # Filtrar y extraer intervalo
    # --------------------------------------------------

    for nombre, (frecuencia_min, frecuencia_max) in bandas.items():

        if frecuencia_min < 0 or frecuencia_max <= frecuencia_min:
            raise ValueError(f"Rango inválido para {nombre}.")

        if frecuencia_max > nyquist:
            raise ValueError(
                f"La frecuencia máxima de {nombre} supera "
                f"Nyquist ({nyquist} Hz)."
            )

        # Diseñar filtro
        if frecuencia_min == 0:
            tipo = "lowpass"
            frecuencias = frecuencia_max

        elif frecuencia_max == nyquist:
            tipo = "highpass"
            frecuencias = frecuencia_min

        else:
            tipo = "bandpass"
            frecuencias = [frecuencia_min, frecuencia_max]

        sos = butter(
            orden,
            frecuencias,
            btype=tipo,
            fs=frecuencia_muestreo,
            output="sos"
        )

        # Filtrar audio completo para evitar artefactos
        # causados por los límites del intervalo seleccionado.
        try:
            senal_filtrada = sosfiltfilt(
                sos,
                audio,
                axis=-1
            )
        except ValueError:
            senal_filtrada = sosfilt(
                sos,
                audio,
                axis=-1
            )

        # Extraer intervalo solicitado
        segmento = senal_filtrada[:, muestra_inicio:muestra_fin].T

        # Guardar WAV
        ruta_salida = directorio / (
            f"{nombre_base}_{nombre}_{inicio}s_{fin}s.wav"
        )

        sf.write(
            ruta_salida,
            segmento,
            frecuencia_muestreo,
            subtype="PCM_16"
        )

        # --------------------------------------------------
        # Calcular intensidad RMS
        # --------------------------------------------------

        muestras_ventana = max(
            1,
            int(ventana_rms * frecuencia_muestreo)
        )

        cantidad_ventanas = len(segmento) // muestras_ventana

        if cantidad_ventanas == 0:
            raise ValueError(
                "El intervalo es demasiado corto para calcular RMS "
                "con la ventana seleccionada."
            )

        segmento_rms = segmento[
            :cantidad_ventanas * muestras_ventana
        ]

        # RMS por ventana y por canal; después se promedian
        # los canales para obtener una sola curva.
        bloques = segmento_rms.reshape(
            cantidad_ventanas,
            muestras_ventana,
            segmento.shape[1]
        )

        rms = np.sqrt(
            np.mean(bloques ** 2, axis=(1, 2))
        )

        tiempo_rms = (
            inicio
            + (np.arange(cantidad_ventanas) + 0.5)
            * ventana_rms
        )

        componentes[nombre] = {
            "senal": segmento,
            "ruta": str(ruta_salida),
            "intensidad_rms": rms,
            "tiempo": tiempo_rms,
            "banda": (frecuencia_min, frecuencia_max)
        }

        print(f"Componente generada: {ruta_salida}")

    # --------------------------------------------------
    # Generar gráfica de intensidades
    # --------------------------------------------------

    fig, ax = plt.subplots(
        len(componentes),
        1,
        figsize=(12, 3 * len(componentes)),
        sharex=True
    )

    if len(componentes) == 1:
        ax = [ax]

    for eje, (nombre, datos) in zip(ax, componentes.items()):

        eje.plot(
            datos["tiempo"],
            datos["intensidad_rms"],
            linewidth=1.2
        )

        eje.set_title(
            f"{nombre.capitalize()} "
            f"({datos['banda'][0]} - {datos['banda'][1]} Hz)"
        )

        eje.set_ylabel("Intensidad RMS")
        eje.grid(True, alpha=0.3)

    ax[-1].set_xlabel("Tiempo (segundos)")

    fig.suptitle(
        f"Intensidad de componentes: {nombre_base}\n"
        f"Intervalo: {inicio:.2f} - {fin:.2f} segundos",
        fontsize=13
    )

    plt.tight_layout()

    ruta_grafica = directorio / (
        f"{nombre_base}_intensidades_{inicio}s_{fin}s.png"
    )

    fig.savefig(
        ruta_grafica,
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()

    print(f"\nGráfica generada: {ruta_grafica}")

    return {
        "componentes": componentes,
        "grafica": str(ruta_grafica),
        "intervalo": (inicio, fin),
        "frecuencia_muestreo": frecuencia_muestreo
    }


# --------------------------------------------------
# Ejemplo de ejecución
# --------------------------------------------------

if __name__ == "__main__":

    resultado = separar_componentes(
        nombre_audio="audio.wav",
        inicio=24,
        fin=34
    )

    print("\nComponentes generadas:")

    for nombre, datos in resultado["componentes"].items():
        print(nombre, "->", datos["ruta"])