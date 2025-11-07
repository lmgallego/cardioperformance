# 🚴 Cycling Performance Analysis Web App

Una aplicación web interactiva construida con Streamlit para analizar datos de rendimiento ciclista. Esta herramienta proporciona análisis en tiempo real de potencia crítica, eficiencia cardiovascular y métricas de rendimiento.

## 🌟 Características

- **Interfaz Web Interactiva**: No requiere conocimientos de programación
- **Análisis de Archivo Único**: Análisis detallado de una sesión individual
- **Análisis de Múltiples Archivos**: Comparación entre múltiples sesiones
- **Visualizaciones Interactivas**: Gráficos de potencia, frecuencia cardíaca y distribuciones
- **Métricas en Tiempo Real**: Cálculo instantáneo de CP, W', rHRI y cuartiles
- **Reporte de Calidad de Datos**: Diagnóstico automático de outliers y datos faltantes
- **Exportación de Resultados**: Descarga resultados en formato CSV

## 📋 Requisitos Previos

- Python 3.10 o superior
- pip (gestor de paquetes de Python)

## 🚀 Instalación

### 1. Clonar el repositorio (si no lo has hecho ya)

```bash
git clone <URL_DEL_REPOSITORIO>
cd cardioperformance
```

### 2. Crear un entorno virtual (recomendado)

```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar las dependencias

```bash
pip install -r requirements.txt
```

## 🎯 Uso

### Iniciar la aplicación web

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

Si no se abre automáticamente, abre tu navegador y navega a esa dirección.

### Detener la aplicación

Presiona `Ctrl+C` en la terminal donde se está ejecutando la aplicación.

## 📊 Modos de Análisis

### 1️⃣ Análisis de Archivo Único

Ideal para analizar una sesión de entrenamiento individual en detalle.

**Pasos:**
1. Selecciona "Single File" en el sidebar
2. Haz clic en "Browse files" para cargar un archivo CSV
3. Espera a que se procese el archivo
4. Explora las siguientes pestañas:
   - **Key Metrics**: Métricas principales (CP, W', R²)
   - **Visualizations**: Gráficos de potencia, FC y distribuciones
   - **Detailed Results**: Tabla completa de resultados
   - **Data Quality**: Reporte de calidad de los datos

**Métricas Clave:**
- **Critical Power (CP)**: Potencia máxima sostenible
- **W' (W-prime)**: Capacidad de trabajo anaeróbico
- **R²**: Calidad del ajuste del modelo
- **rHRI por cuartil**: Eficiencia cardiovascular

### 2️⃣ Análisis de Múltiples Archivos

Perfecto para comparar múltiples sesiones de entrenamiento.

**Pasos:**
1. Selecciona "Multiple Files" en el sidebar
2. Haz clic en "Browse files" y selecciona múltiples archivos CSV
3. Espera a que se procesen todos los archivos
4. Explora las siguientes pestañas:
   - **Summary Statistics**: Estadísticas agregadas de todas las sesiones
   - **Comparison**: Gráficos comparativos entre sesiones
   - **Detailed Results**: Tabla completa con todas las sesiones

**Funcionalidades:**
- Comparación visual de CP y W' entre archivos
- Estadísticas agregadas (media, desviación estándar)
- Descarga de resultados en CSV
- Comparación de grupos (Top 5 vs No Top 5)

## 📁 Formato de Datos

Los archivos CSV deben tener las siguientes columnas:

| Columna | Descripción | Unidad |
|---------|-------------|--------|
| `time` | Tiempo transcurrido | segundos |
| `power` o `watts` | Potencia de salida | vatios (W) |
| `heart_rate` o `heartrate` | Frecuencia cardíaca | latidos por minuto (bpm) |

### Ejemplo de formato CSV:

```csv
time,power,heart_rate
0,150,120
1,160,125
2,155,123
3,165,128
4,170,130
```

## 📈 Interpretación de Resultados

### Critical Power (CP)
La potencia máxima que puedes mantener en estado casi-estacionario sin fatiga. Valores más altos indican mejor capacidad aeróbica.

### W' (W-prime)
La cantidad finita de trabajo que se puede realizar por encima de la potencia crítica. Representa la capacidad anaeróbica.

### rHRI (Relative Heart Rate Increase)
Métrica de eficiencia cardiovascular que cuantifica qué tan eficientemente tu sistema cardiovascular responde al ejercicio.

**Fórmula:** `rHRI = (Derivada de FC) / Potencia`

**Interpretación:** Valores más bajos indican mejor eficiencia cardiovascular.

### Análisis por Cuartiles

Los datos se dividen en 4 cuartiles basados en el porcentaje de la potencia crítica:

- **Q1**: 25% inferior (potencia baja)
- **Q2**: 25-50% (potencia moderada-baja)
- **Q3**: 50-75% (potencia moderada-alta)
- **Q4**: 25% superior (potencia alta)

## 🔧 Configuración Avanzada

### Cambiar el puerto de la aplicación

```bash
streamlit run app.py --server.port 8502
```

### Cambiar el tema

Crea un archivo `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Aumentar el límite de tamaño de archivo

Por defecto, Streamlit limita los archivos a 200 MB. Para cambiarlo:

```bash
streamlit run app.py --server.maxUploadSize 500
```

O en `.streamlit/config.toml`:

```toml
[server]
maxUploadSize = 500
```

## 🐛 Solución de Problemas

### La aplicación no se inicia

1. Verifica que todas las dependencias estén instaladas:
   ```bash
   pip install -r requirements.txt
   ```

2. Verifica la versión de Python:
   ```bash
   python --version  # Debe ser >= 3.10
   ```

### Error al cargar archivos

1. Verifica que el archivo CSV tenga las columnas requeridas
2. Asegúrate de que el archivo esté codificado en UTF-8
3. Verifica que no haya caracteres especiales en el nombre del archivo

### Gráficos no se muestran

1. Verifica que matplotlib esté instalado:
   ```bash
   pip install matplotlib
   ```

2. Si estás en un servidor remoto, asegúrate de que el puerto esté accesible

### Errores de memoria

Si tienes archivos muy grandes:

1. Aumenta el límite de memoria de Streamlit
2. Considera procesar archivos más pequeños
3. Reduce la ventana de rolling mean en `cycling_analysis.py`

## 📚 Recursos Adicionales

- [Documentación de Streamlit](https://docs.streamlit.io/)
- [Documentación del Proyecto Principal](README.md)
- [Repositorio de GitHub](https://github.com/anthropics/claude-code)

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para cambios importantes:

1. Haz fork del repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Notas de la Versión Web

### Versión 1.0.0 (2025)

**Características iniciales:**
- ✅ Análisis de archivo único
- ✅ Análisis de múltiples archivos
- ✅ Visualizaciones interactivas
- ✅ Reporte de calidad de datos
- ✅ Exportación de resultados
- ✅ Interfaz responsive
- ✅ Temas personalizables

**Mejoras futuras planeadas:**
- 📅 Seguimiento de progreso temporal
- 📅 Comparación con sesiones anteriores
- 📅 Exportación de gráficos en PNG/PDF
- 📅 Análisis de tendencias a largo plazo
- 📅 Recomendaciones de entrenamiento basadas en IA

## 📧 Contacto

Para preguntas, problemas o sugerencias, por favor abre un issue en el repositorio de GitHub.

---

**Construido con ❤️ usando Streamlit 1.51.0**
