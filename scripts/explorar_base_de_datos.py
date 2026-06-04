import sys
from pathlib import Path

RUTA_PROYECTO = Path(__file__).resolve().parents[1]
sys.path.append(str(RUTA_PROYECTO))

import statsmodels.api as sm

from src.carga_datos import CargaDatos
from src.models.main_model import BusquedaModelo

COLUMNAS_NUMERICAS = [
    "Age",
    "MP",
    "Starts",
    "Min",
    "Gls",
    "Ast",
    "Sh",
    "SoT",
    "xG",
    "xAG",
    "Cmp",
    "Att",
    "PrgP",
    "Tkl",
    "Int",
    "90s",
    "G+A",
    "Sh/90",
    "Cmp%",
    "KP",
    "SCA",
    "PrgC",
    "Carries",
    "CrdY",
    "CrdR",
]


def main():
    ruta_datos = RUTA_PROYECTO / "data" / "jugadores_2024_2025.csv"
    cargador = CargaDatos(ruta_datos)

    cargador.cargar_datos()
    cargador.mostrar_dimensiones()

    print()
    cargador.mostrar_primeras_filas(n=5)

    print()
    cargador.mostrar_informacion()

    print()
    cargador.mostrar_valores_faltantes(n=10)

    print()
    cargador.mostrar_resumen_numerico(columnas=COLUMNAS_NUMERICAS)

    print()
    busqueda = BusquedaModelo(cargador.obtener_datos())

    modelo = busqueda.buscar_mejor_modelo()

    print("\n")
    print(modelo.summary())

    modelo.graficar()


if __name__ == "__main__":
    main()
