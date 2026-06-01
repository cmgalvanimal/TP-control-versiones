from pathlib import Path

import pandas as pd


class CargaDatos:
    """Carga un archivo CSV y permite realizar una exploración inicial."""

    def __init__(self, ruta_datos):
        """
        Parameters
        ----------
        ruta_datos : str or pathlib.Path
            Ruta al archivo CSV.
        """
        self.ruta_datos = Path(ruta_datos)
        self.datos = None

    def cargar_datos(self):
        """Carga el archivo CSV y devuelve su contenido como DataFrame."""
        if not self.ruta_datos.exists():
            raise FileNotFoundError(f"No se encontró el archivo: {self.ruta_datos}")

        self.datos = pd.read_csv(self.ruta_datos)
        return self.datos

    def mostrar_dimensiones(self):
        """Muestra la cantidad de filas y columnas del dataset."""
        self._verificar_datos_cargados()

        print("Dimensiones del dataset:")
        print(f"- Filas: {self.datos.shape[0]}")
        print(f"- Columnas: {self.datos.shape[1]}")

    def mostrar_columnas(self):
        """Muestra los nombres de todas las columnas disponibles."""
        self._verificar_datos_cargados()

        print("Columnas disponibles:")
        for columna in self.datos.columns:
            print(f"- {columna}")

    def mostrar_primeras_filas(self, n=5, columnas=None):
        """
        Muestra las primeras filas del dataset.

        Parameters
        ----------
        n : int
            Cantidad de filas a mostrar.
        columnas : list[str] or None
            Columnas que se desean visualizar. Si no se indican, se muestran
            todas las columnas.
        """
        self._verificar_datos_cargados()
        datos_a_mostrar = self._seleccionar_columnas(columnas)

        print(f"Primeras {n} filas:")
        print(datos_a_mostrar.head(n).to_string(index=False))

    def mostrar_informacion(self, columnas=None):
        """
        Muestra los tipos de datos y la cantidad de valores no nulos.

        Parameters
        ----------
        columnas : list[str] or None
            Columnas que se desean visualizar. Si no se indican, se muestran
            todas las columnas.
        """
        self._verificar_datos_cargados()
        datos_a_mostrar = self._seleccionar_columnas(columnas)

        print("Información general del dataset:")
        datos_a_mostrar.info()

    def mostrar_valores_faltantes(self, n=10):
        """
        Muestra las columnas que tienen más valores faltantes.

        Parameters
        ----------
        n : int
            Cantidad máxima de columnas a mostrar.
        """
        self._verificar_datos_cargados()

        valores_faltantes = self.datos.isna().sum()
        valores_faltantes = valores_faltantes[valores_faltantes > 0]
        valores_faltantes = valores_faltantes.sort_values(ascending=False)

        print(f"Columnas con más valores faltantes (máximo {n}):")
        if valores_faltantes.empty:
            print("No se encontraron valores faltantes.")
        else:
            print(valores_faltantes.head(n).to_string())

    def mostrar_resumen_numerico(self, columnas=None):
        """
        Muestra estadísticas descriptivas de las variables numéricas.

        Parameters
        ----------
        columnas : list[str] or None
            Columnas que se desean visualizar. Si no se indican, se incluyen
            todas las columnas numéricas.
        """
        self._verificar_datos_cargados()
        datos_a_mostrar = self._seleccionar_columnas(columnas)

        print("Resumen de variables numéricas:")
        print(datos_a_mostrar.describe().round(2).to_string())

    def obtener_datos(self):
        """Devuelve el DataFrame cargado."""
        self._verificar_datos_cargados()
        return self.datos

    def seleccionar_columnas(self, columnas):
        """Conserva únicamente las columnas indicadas y devuelve el resultado."""
        self._verificar_datos_cargados()
        self.datos = self._seleccionar_columnas(columnas).copy()
        return self.datos

    def guardar_datos(self, ruta_salida):
        """Guarda el DataFrame actual en un archivo CSV."""
        self._verificar_datos_cargados()
        ruta_salida = Path(ruta_salida)
        self.datos.to_csv(ruta_salida, index=False)

    def _seleccionar_columnas(self, columnas):
        """Devuelve las columnas solicitadas y verifica que existan."""
        if columnas is None:
            return self.datos

        columnas_inexistentes = set(columnas) - set(self.datos.columns)
        if columnas_inexistentes:
            raise ValueError(
                "No se encontraron las siguientes columnas: "
                f"{sorted(columnas_inexistentes)}"
            )

        return self.datos[columnas]

    def _verificar_datos_cargados(self):
        """Verifica que el CSV haya sido cargado antes de utilizarlo."""
        if self.datos is None:
            raise ValueError("Primero debe cargar los datos usando cargar_datos().")
