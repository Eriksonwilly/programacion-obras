# Aplicación PERT-CPM para Control de Obras

## Descripción
Aplicación web desarrollada en Streamlit para el análisis y control de proyectos de construcción utilizando los métodos PERT y CPM (Critical Path Method).

## Características
- Análisis PERT con tiempos optimista, probable y pesimista
- Cálculo de ruta crítica (CPM)
- Diagrama de red interactivo
- Cronograma de ejecución (Gantt)
- Cálculo de probabilidades de conclusión
- Exportación a PDF
- Control de materiales por actividad

## Instalación

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Ejecutar la aplicación
```bash
streamlit run "APP1 analsisi respaldo.py"
```

## Uso

1. **Ingreso de datos PERT**: Complete los tiempos optimista, probable y pesimista para cada actividad
2. **Cálculo automático**: Presione "Calcular PERT y CPM" para obtener resultados
3. **Visualización**: Revise el diagrama de red, tablas y cronogramas
4. **Exportación**: Descargue los resultados en formato PDF

## Estructura del Proyecto
- **Actividades incluidas**: 20 actividades de construcción organizadas por especialidad
- **Especialidades**: Estructuras, Instalaciones, Acabados
- **Duración total estimada**: Variable según tiempos PERT ingresados

## Solución de Problemas

### Error de Memoria
Si aparece un error de memoria:
1. Reduzca los valores de entrada
2. Reinicie la aplicación
3. Cierre otras aplicaciones que consuman memoria

### Dependencias faltantes
Si hay errores de importación:
```bash
pip install --upgrade streamlit pandas networkx matplotlib numpy reportlab Pillow plotly
```

## Información del Proyecto
- **Proyecto**: Construcción de Vivienda de Dos Plantas + Azotea
- **Ubicación**: Chiclayo, Lambayeque
- **Empresa**: CONSORCIO DEJ
- **Elaborado por**: Ing. Civil UNI (20 años de experiencia en edificación) 