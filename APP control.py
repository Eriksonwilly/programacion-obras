import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from datetime import datetime, timedelta
import numpy as np
import math
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A2, landscape
from reportlab.lib.utils import ImageReader
from PIL import Image
import gc
import os
import importlib
import importlib.util
import sys

def check_dependencies():
    """Verifica las dependencias principales con mayor flexibilidad en versiones y manejo especial para kaleido"""
    required_packages = {
        'streamlit': {'min': '1.22.0'},
        'pandas': {'min': '1.5.0'},
        'matplotlib': {'min': '3.6.0'},
        'networkx': {'min': '3.0'},
        'numpy': {'min': '1.23.0'},
        'PIL': {'min': '9.4.0'},
        'reportlab': {'min': '3.6.0'},
        'plotly': {'min': '5.11.0'},
        'dateutil': {'min': '2.8.2'},
        'packaging': {'min': '23.0'}
    }
    missing = []
    outdated = []
    # Verificar kaleido primero de manera especial
    try:
        import kaleido
        from packaging import version
        current_version = getattr(kaleido, '__version__', '0.2.1')
        if version.parse(current_version) < version.parse('0.2.1'):
            outdated.append(f"kaleido (requerido: >= 0.2.1, instalado: {current_version})")
    except ImportError:
        missing.append('kaleido')
    except Exception:
        missing.append('kaleido')
    # Verificar el resto de paquetes
    for package, reqs in required_packages.items():
        try:
            module = importlib.import_module(package)
            current_version = getattr(module, '__version__', '0.0.0')
            try:
                from packaging import version
                if version.parse(current_version) < version.parse(reqs['min']):
                    outdated.append(f"{package} (requerido: >= {reqs['min']}, instalado: {current_version})")
            except ImportError:
                if package != 'packaging':
                    missing.append('packaging')
        except ImportError:
            display_name = {
                'PIL': 'Pillow',
                'dateutil': 'python-dateutil'
            }.get(package, package)
            if display_name not in missing:
                missing.append(display_name)
    return missing, outdated

# Configurar límites de PIL para evitar DecompressionBombError
Image.MAX_IMAGE_PIXELS = None

st.set_page_config(
    page_title="CPM-PERT Control de Obras - Vivienda Chiclayo",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_project_data():
    activities = {
        'A01': {'name': 'Limpieza y replanteo', 'duration': 1, 'predecessors': [], 'category': 'Estructuras'},
        'A02': {'name': 'Excavación de zanjas para cimentación', 'duration': 2, 'predecessors': ['A01'], 'category': 'Estructuras'},
        'B01': {'name': 'Colocación de concreto pobre', 'duration': 1, 'predecessors': ['A02'], 'category': 'Estructuras'},
        'B02': {'name': 'Armado de zapata y cimiento corrido', 'duration': 3, 'predecessors': ['B01'], 'category': 'Estructuras'},
        'B03': {'name': 'Vaciado de concreto fc=175 kg/cm²', 'duration': 1, 'predecessors': ['B02'], 'category': 'Estructuras'},
        'B04': {'name': 'Curado de cimentación', 'duration': 2, 'predecessors': ['B03'], 'category': 'Estructuras'},
        'C01': {'name': 'Encofrado y armado de sobrecimiento', 'duration': 2, 'predecessors': ['B04'], 'category': 'Estructuras'},
        'C02': {'name': 'Vaciado de concreto fc=210 kg/cm²', 'duration': 1, 'predecessors': ['C01'], 'category': 'Estructuras'},
        'C03': {'name': 'Desencofrado y curado', 'duration': 2, 'predecessors': ['C02'], 'category': 'Estructuras'},
        'D01': {'name': 'Levantado de muros (bloques de concreto)', 'duration': 5, 'predecessors': ['C03'], 'category': 'Estructuras'},
        'D02': {'name': 'Colocación de vigas soleras', 'duration': 2, 'predecessors': ['D01'], 'category': 'Estructuras'},
        'E01': {'name': 'Encofrado de techo aligerado', 'duration': 3, 'predecessors': ['D02'], 'category': 'Estructuras'},
        'E02': {'name': 'Armado de viguetas y losa', 'duration': 2, 'predecessors': ['E01'], 'category': 'Estructuras'},
        'E03': {'name': 'Vaciado de concreto fc=210 kg/cm²', 'duration': 1, 'predecessors': ['E02'], 'category': 'Estructuras'},
        'E04': {'name': 'Curado del techo', 'duration': 7, 'predecessors': ['E03'], 'category': 'Estructuras'},
        'F01': {'name': 'Instalaciones eléctricas (embutidas)', 'duration': 3, 'predecessors': ['E04'], 'category': 'Instalaciones'},
        'F02': {'name': 'Instalaciones sanitarias', 'duration': 2, 'predecessors': ['E04'], 'category': 'Instalaciones'},
        'G01': {'name': 'Tarrajeo de muros interiores/exteriores', 'duration': 4, 'predecessors': ['F01', 'F02'], 'category': 'Acabados'},
        'G02': {'name': 'Pisos de porcelanato', 'duration': 3, 'predecessors': ['G01'], 'category': 'Acabados'},
        'G03': {'name': 'Pintura', 'duration': 2, 'predecessors': ['G02'], 'category': 'Acabados'}
    }
    return activities

def get_materials_data():
    # Genera materiales por actividad (puedes personalizar)
    return pd.DataFrame([
        {'Actividad': k, 'Material': f'Material {k}', 'Cantidad': np.random.randint(1, 10), 'Fecha Necesaria': ''}
        for k in get_project_data().keys()
    ])

def calculate_pert(activities, pert_inputs):
    try:
        pert_results = {}
        for code, data in activities.items():
            if code in pert_inputs:
                o = pert_inputs[code]['optimista']
                m = pert_inputs[code]['probable']
                p = pert_inputs[code]['pesimista']
                
                # Validar que los tiempos sean válidos
                if o <= m <= p and o > 0:
                    te = (o + 4*m + p) / 6
                    var = ((p - o) / 6) ** 2
                    pert_results[code] = {'te': te, 'var': var, 'o': o, 'm': m, 'p': p}
                else:
                    # Usar valores por defecto si los tiempos no son válidos
                    te = data['duration']
                    var = 1.0
                    pert_results[code] = {'te': te, 'var': var, 'o': o, 'm': m, 'p': p}
            else:
                # Usar duración por defecto si no hay datos PERT
                te = data['duration']
                var = 1.0
                pert_results[code] = {'te': te, 'var': var, 'o': te, 'm': te, 'p': te}
        
        return pert_results
    except Exception as e:
        st.error(f"Error en cálculo PERT: {str(e)}")
        # Retornar valores por defecto en caso de error
        return {code: {'te': data['duration'], 'var': 1.0, 'o': data['duration'], 
                      'm': data['duration'], 'p': data['duration']} 
                for code, data in activities.items()}

def calculate_cpm(activities, pert_results=None):
    try:
        es, ef, ls, lf = {}, {}, {}, {}
        
        # Inicializar valores
        for activity in activities:
            es[activity] = 0
            ef[activity] = 0
            ls[activity] = 0
            lf[activity] = 0
        
        # Forward pass - calcular ES y EF
        for activity in activities:
            dur = pert_results[activity]['te'] if pert_results else activities[activity]['duration']
            if not activities[activity]['predecessors']:
                es[activity] = 0
            else:
                es[activity] = max(ef[pred] for pred in activities[activity]['predecessors'] if pred in ef)
            ef[activity] = es[activity] + dur
        
        # Encontrar actividades finales
        final_activities = []
        for activity in activities:
            is_final = True
            for other_activity in activities:
                if activity in activities[other_activity]['predecessors']:
                    is_final = False
                    break
            if is_final:
                final_activities.append(activity)
        
        max_ef = max(ef.values()) if ef else 0
        
        # Backward pass - calcular LS y LF
        for activity in activities:
            dur = pert_results[activity]['te'] if pert_results else activities[activity]['duration']
            if activity in final_activities:
                lf[activity] = max_ef
            else:
                successors = []
                for other_activity in activities:
                    if activity in activities[other_activity]['predecessors']:
                        successors.append(other_activity)
                if successors:
                    lf[activity] = min(ls[succ] for succ in successors if succ in ls)
                else:
                    lf[activity] = max_ef
            ls[activity] = lf[activity] - dur
        
        # Calcular holgura y ruta crítica
        slack = {}
        critical_path = []
        for activity in activities:
            slack[activity] = ls[activity] - es[activity]
            if abs(slack[activity]) < 1e-6:
                critical_path.append(activity)
        
        return {
            'es': es, 'ef': ef, 'ls': ls, 'lf': lf, 
            'slack': slack, 'critical_path': critical_path,
            'total_duration': max_ef
        }
    except Exception as e:
        st.error(f"Error en cálculo CPM: {str(e)}")
        # Retornar valores por defecto en caso de error
        return {
            'es': {k: 0 for k in activities},
            'ef': {k: 0 for k in activities},
            'ls': {k: 0 for k in activities},
            'lf': {k: 0 for k in activities},
            'slack': {k: 0 for k in activities},
            'critical_path': [],
            'total_duration': 0
        }

def draw_pert_cpm_diagram(activities, cpm_results, pert_results, show_table=False, fig_width=10, fig_height=7, node_size=2000, max_nodes=30):
    try:
        # Limpiar cualquier figura previa
        plt.close('all')
        
        # Crear el grafo
        G = nx.DiGraph()
        for activity in activities:
            G.add_node(activity)
        for activity in activities:
            for pred in activities[activity]['predecessors']:
                G.add_edge(pred, activity)

        # Configurar el layout robusto (sin dependencias externas)
        try:
            # Usar layout spring mejorado como opción principal
            pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42)
        except Exception as layout_error:
            # Si falla, usar layout circular como respaldo
            pos = nx.circular_layout(G)
        
        # Crear figura con tamaño optimizado para Streamlit
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)
        plt.rcParams['font.size'] = 8
        plt.rcParams['axes.linewidth'] = 0.8
        
        # Dibujar aristas con estilo mejorado
        for edge in G.edges():
            is_critical = edge[0] in cpm_results['critical_path'] and edge[1] in cpm_results['critical_path']
            edge_color = '#FF0000' if is_critical else '#333333'
            edge_width = 2.0 if is_critical else 1.0
            edge_style = 'solid' if is_critical else 'dashed'
            
            nx.draw_networkx_edges(G, pos, edgelist=[edge], 
                                 edge_color=edge_color, 
                                 width=edge_width, 
                                 style=edge_style,
                                 ax=ax, 
                                 arrows=True, 
                                 arrowstyle='->',
                                 arrowsize=12)
        
        # Dibujar nodos con información simplificada
        node_colors = []
        edge_colors = []
        labels = {}
        
        for node in G.nodes():
            te = pert_results[node]['te'] if pert_results else activities[node]['duration']
            is_critical = node in cpm_results['critical_path']
            
            # Configurar colores según ruta crítica
            node_colors.append('#FF6B6B' if is_critical else '#6BB9FF')
            edge_colors.append('#FF0000' if is_critical else '#0066CC')
            
            # Crear etiqueta simplificada
            labels[node] = f"{node}\n{activities[node]['name']}\n{te:.1f}d"
        
        # Dibujar todos los nodos de una vez para mejor rendimiento
        # Asegurarse de que node_colors es una lista de strings
        node_colors = [str(c) for c in node_colors]
        nx.draw_networkx_nodes(G, pos, node_size=node_size, 
                              node_color=node_colors, 
                              edgecolors=edge_colors,
                              linewidths=1.5,
                              ax=ax)  # type: ignore
        
        # Dibujar etiquetas
        nx.draw_networkx_labels(G, pos, labels, font_size=8, 
                               font_family='sans-serif',
                               ax=ax)
        
        # Añadir información adicional en posiciones estratégicas
        for node in G.nodes():
            x, y = pos[node]
            es = cpm_results['es'][node]
            ef = cpm_results['ef'][node]
            ls = cpm_results['ls'][node]
            lf = cpm_results['lf'][node]
            slack = cpm_results['slack'][node]
            
            # Mostrar solo información clave alrededor del nodo
            ax.text(x, y+0.1, f"ES:{es:.0f} | EF:{ef:.0f}", 
                   fontsize=7, ha='center', va='bottom', color='#006600')
            ax.text(x, y-0.1, f"LS:{ls:.0f} | LF:{lf:.0f}", 
                   fontsize=7, ha='center', va='top', color='#CC6600')
            
            if slack > 0.01:
                ax.text(x, y-0.15, f"Holgura: {slack:.0f}", 
                       fontsize=7, ha='center', va='top', color='#660066')

        # Títulos y leyenda optimizados
        ax.set_title("DIAGRAMA DE RED PERT-CPM", fontsize=12, pad=20)
        ax.text(0.5, 1.02, "Proyecto: Construcción de Vivienda de Dos Plantas + Azotea", 
               fontsize=9, ha='center', va='bottom', transform=ax.transAxes)
        
        # Leyenda simplificada
        legend_elements = [
            plt.Line2D([0], [0], color='#FF0000', lw=2, label='Ruta Crítica'),
            plt.Line2D([0], [0], color='#333333', lw=1, linestyle='dashed', label='Ruta Normal'),
            plt.Line2D([0], [0], marker='o', color='w', label='Actividad Crítica',
                      markerfacecolor='#FF6B6B', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Actividad Normal',
                      markerfacecolor='#6BB9FF', markersize=10)
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1.15, 1), fontsize=8, framealpha=0.9)
        
        ax.axis('off')
        plt.tight_layout()
        
        # Guardar imagen con calidad optimizada
        img_path = "diagrama_pert_cpm.png"
        fig.savefig(img_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return img_path
        
    except Exception as e:
        st.error(f"Error al generar diagrama: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return None

def export_to_pdf(img_path, table_df, gantt_img_path=None):
    try:
        gc.collect()
        pdf_buf = io.BytesIO()
        c = canvas.Canvas(pdf_buf, pagesize=landscape(A2))
        from datetime import datetime
        # Configurar márgenes y estilos
        left_margin = 40
        right_margin = 40
        top_margin = 750
        section_spacing = 30
        # Encabezado del reporte
        c.setFont("Helvetica-Bold", 16)
        c.drawString(left_margin, top_margin, "REPORTE PERT-CPM - CONTROL DE OBRAS")
        c.setFont("Helvetica", 12)
        c.drawString(left_margin, top_margin - 20, "Proyecto: Construcción de Vivienda de Dos Plantas + Azotea")
        c.drawString(left_margin, top_margin - 40, "Ubicación: Chiclayo, Lambayeque | Empresa: CONSORCIO DEJ")
        c.drawString(left_margin, top_margin - 60, "Elaborado por: Erikson Florez Valdivia UNA - Puno Ing Civil")
        # Sección 1: Diagrama de Red
        current_y = top_margin - 90
        c.setFont("Helvetica-Bold", 14)
        c.drawString(left_margin, current_y, "1. Diagrama de Red PERT-CPM")
        current_y -= 20
        try:
            if img_path and os.path.exists(img_path):
                img = Image.open(img_path)
                img_width = 1000
                img_height = int(img.height * (img_width / img.width))
                resized_img_path = "diagrama_pert_cpm_resized.png"
                img.resize((img_width, img_height), Image.LANCZOS).save(resized_img_path)
                c.drawImage(resized_img_path, left_margin, current_y - img_height, 
                          width=img_width, height=img_height, mask='auto')
                current_y -= img_height + section_spacing
                os.remove(resized_img_path)
            else:
                c.drawString(left_margin, current_y, "No se pudo cargar el diagrama de red.")
                current_y -= section_spacing
        except Exception as e:
            c.drawString(left_margin, current_y, f"Error al insertar diagrama de red: {str(e)}")
            current_y -= section_spacing
        # Sección 2: Diagrama de Gantt
        if gantt_img_path and os.path.exists(gantt_img_path):
            c.setFont("Helvetica-Bold", 14)
            c.drawString(left_margin, current_y, "2. Diagrama de Gantt - Cronograma de Ejecución")
            current_y -= 20
            try:
                gantt_img = Image.open(gantt_img_path)
                gantt_width = 1000
                gantt_height = int(gantt_img.height * (gantt_width / gantt_img.width))
                max_gantt_height = 500
                if gantt_height > max_gantt_height:
                    gantt_height = max_gantt_height
                    gantt_width = int(gantt_img.width * (gantt_height / gantt_img.height))
                resized_gantt_path = "diagrama_gantt_resized.png"
                gantt_img.resize((gantt_width, gantt_height), Image.LANCZOS).save(resized_gantt_path)
                c.setFillColorRGB(0.95, 0.95, 0.95)
                c.rect(left_margin-5, current_y - gantt_height -5, 
                      gantt_width+10, gantt_height+10, fill=1, stroke=0)
                c.setFillColorRGB(0, 0, 0)
                c.drawImage(resized_gantt_path, left_margin, current_y - gantt_height, 
                          width=gantt_width, height=gantt_height, mask='auto')
                current_y -= gantt_height + section_spacing + 20
                os.remove(resized_gantt_path)
            except Exception as e:
                c.drawString(left_margin, current_y, f"Error al insertar diagrama Gantt: {str(e)}")
                current_y -= section_spacing
        # Sección 3: Tabla de actividades
        c.setFont("Helvetica-Bold", 14)
        c.drawString(left_margin, current_y, "3. Tabla de Actividades y Resultados PERT")
        current_y -= 20
        col_widths = [60, 200, 80, 60, 60, 60, 60, 100]
        row_height = 18
        headers = list(table_df.columns)
        c.setFillColorRGB(0.8, 0.8, 0.8)
        c.rect(left_margin, current_y - row_height + 5, sum(col_widths), row_height, fill=1, stroke=0)
        c.setFillColorRGB(0, 0, 0)
        c.setFont("Helvetica-Bold", 10)
        for i, h in enumerate(headers):
            c.drawString(left_margin + sum(col_widths[:i]), current_y, str(h))
        current_y -= row_height
        c.setFont("Helvetica", 9)
        for idx, row in table_df.iterrows():
            if current_y < 100:
                c.showPage()
                current_y = top_margin - 40
                c.setFillColorRGB(0.8, 0.8, 0.8)
                c.rect(left_margin, current_y - row_height + 5, sum(col_widths), row_height, fill=1, stroke=0)
                c.setFillColorRGB(0, 0, 0)
                c.setFont("Helvetica-Bold", 10)
                for i, h in enumerate(headers):
                    c.drawString(left_margin + sum(col_widths[:i]), current_y, str(h))
                current_y -= row_height
                c.setFont("Helvetica", 9)
            if idx % 2 == 0:
                c.setFillColorRGB(0.95, 0.95, 0.95)
                c.rect(left_margin, current_y - row_height + 5, sum(col_widths), row_height, fill=1, stroke=0)
                c.setFillColorRGB(0, 0, 0)
            for i, val in enumerate(row):
                text_val = str(round(val,2)) if isinstance(val, float) else str(val)
                if i in [2,3,4,5,6]:
                    c.drawCentredString(left_margin + sum(col_widths[:i]) + col_widths[i]/2, 
                                      current_y, text_val)
                else:
                    c.drawString(left_margin + sum(col_widths[:i]), current_y, text_val)
            current_y -= row_height
        c.showPage()
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(A2[0]/2, 700, "Resumen del Proyecto")
        c.setFont("Helvetica", 12)
        c.drawString(100, 650, f"Duración total del proyecto: {table_df['Duración Esperada'].sum():.1f} días")
        y_pos = 620
        for esp in table_df['Especialidad'].unique():
            duracion_esp = table_df[table_df['Especialidad']==esp]['Duración Esperada'].sum()
            c.drawString(100, y_pos, f"{esp}: {duracion_esp:.1f} días")
            y_pos -= 30
        c.setFont("Helvetica-Bold", 12)
        c.drawString(100, y_pos-40, "Documento generado el: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        c.save()
        pdf_buf.seek(0)
        try:
            with open("PERT_CPM_RESULTADOS.pdf", "wb") as f:
                f.write(pdf_buf.getvalue())
        except Exception as e:
            st.warning(f"No se pudo guardar el archivo PDF automáticamente en disco: {str(e)}")
        return pdf_buf
    except Exception as e:
        st.error(f"Error al exportar PDF: {str(e)}")
        return None

def main():
    try:
        # Verificar dependencias
        missing, outdated = check_dependencies()
        # Manejo especial para kaleido
        if 'kaleido' in missing:
            st.warning("""
            **Atención:** Se requiere kaleido para la generación de gráficos.
            Este paquete necesita instalación especial debido a dependencias binarias.
            """)
            if st.button("Instalar kaleido automáticamente (recomendado)"):
                try:
                    import subprocess
                    import sys
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "kaleido==0.2.1", "--no-deps", "--force-reinstall"],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        st.success("""
                        ¡kaleido instalado correctamente! 
                        Por favor recargue la página (F5 o botón de recargar del navegador).
                        """)
                    else:
                        st.error(f"Error al instalar kaleido: {result.stderr}")
                        st.info("""
                        Si falla la instalación automática, pruebe manualmente en su terminal:
                        pip install kaleido==0.2.1 --no-deps --force-reinstall
                        """)
                    return
                except Exception as e:
                    st.error(f"Error en la instalación: {str(e)}")
                    return
        if missing and 'kaleido' not in missing:
            st.error(f"Paquetes faltantes: {', '.join(missing)}")
            st.info("""
            Puede instalar los paquetes faltantes con:
            pip install Pillow python-dateutil
            O desde el archivo requirements.txt:
            pip install -r requirements.txt
            """)
            return
        if outdated:
            st.warning("Algunas dependencias tienen versiones diferentes a las recomendadas:")
            for item in outdated:
                st.warning(item)
            st.info("Para mejor compatibilidad, actualice los paquetes con:")
            st.code("pip install --upgrade -r requirements.txt")
            if not st.checkbox("Continuar con las versiones actuales (puede haber incompatibilidades)"):
                return
        
        activities = get_project_data()
        st.title("Presentación Integral PERT-CPM y Control de Obras")
        st.markdown("""
        <b>Proyecto:</b> Construcción de Vivienda de Dos Plantas + Azotea  
        <b>Ubicación:</b> Chiclayo, Lambayeque  
        <b>Elaborado por:</b> Erikson Florez Valdivia UNA - Puno Ing Civil - Control Obra y Edyficaciones  
        <b>Empresa:</b> CONSORCIO DEJ
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        st.subheader("1. Ingrese los tiempos PERT para cada actividad")
        pert_inputs = {}
        
        with st.form("pert_form"):
            cols = st.columns(4)
            for code, data in activities.items():
                with cols[hash(code)%4]:
                    st.markdown(f"**{code}: {data['name']}**")
                    o = st.number_input(f"Optimista ({code})", min_value=1.0, value=float(data['duration']), key=f"o_{code}")
                    m = st.number_input(f"Más probable ({code})", min_value=1.0, value=float(data['duration']), key=f"m_{code}")
                    p = st.number_input(f"Pesimista ({code})", min_value=1.0, value=float(data['duration']+1), key=f"p_{code}")
                    pert_inputs[code] = {'optimista': o, 'probable': m, 'pesimista': p}
            
            submitted = st.form_submit_button("Calcular PERT y CPM")
        
        if submitted:
            try:
                # Limpiar memoria antes de comenzar cálculos
                gc.collect()
                plt.close('all')
                
                # Verificar que todos los datos estén completos
                if not pert_inputs or len(pert_inputs) != len(activities):
                    st.error("Por favor complete todos los tiempos PERT antes de calcular.")
                    return
                
                pert_results = calculate_pert(activities, pert_inputs)
                cpm_results = calculate_cpm(activities, pert_results)
                st.success("¡Cálculo realizado!")
                
                st.subheader("2. Diagrama de Red PERT-CPM")
                try:
                    # Tamaño optimizado para Streamlit
                    img_path = draw_pert_cpm_diagram(activities, cpm_results, pert_results, 
                                                   fig_width=10, fig_height=7, node_size=2000)
                    
                    if img_path:
                        # Mostrar imagen con ancho ajustado
                        st.image(img_path, caption="Diagrama de Red PERT-CPM", width=900)
                        
                        # Botón de descarga
                        with open(img_path, "rb") as img_file:
                            st.download_button(
                                "Descargar Diagrama como PNG", 
                                data=img_file, 
                                file_name="diagrama_pert_cpm.png", 
                                mime="image/png"
                            )
                    else:
                        st.warning("No se pudo generar el diagrama.")
                        
                except Exception as diagram_error:
                    st.error(f"Error al mostrar diagrama: {str(diagram_error)}")
                
                st.subheader("3. Tabla de Actividades y Resultados PERT")
                try:
                    table_data = []
                    for k, v in activities.items():
                        if k in pert_results:
                            row = [
                                k, 
                                v['name'], 
                                pert_results[k]['te'], 
                                pert_results[k]['o'], 
                                pert_results[k]['m'], 
                                pert_results[k]['p'], 
                                pert_results[k]['var'], 
                                v['category']
                            ]
                            table_data.append(row)
                    
                    table_df = pd.DataFrame(table_data, 
                                          columns=["Código", "Actividad", "Duración Esperada", "Optimista", 
                                                  "Más Probable", "Pesimista", "Varianza", "Especialidad"])
                    st.dataframe(table_df, use_container_width=True)
                except Exception as table_error:
                    st.error(f"Error al generar tabla: {str(table_error)}")
                
                st.subheader("4. Probabilidad de Conclusión del Proyecto y por Especialidad")
                try:
                    for esp in table_df['Especialidad'].unique():
                        esp_acts = table_df[table_df['Especialidad']==esp]['Código'].tolist()
                        te_esp = sum(pert_results[k]['te'] for k in esp_acts if k in pert_results)
                        var_esp = sum(pert_results[k]['var'] for k in esp_acts if k in pert_results)
                        std_esp = np.sqrt(var_esp) if var_esp > 0 else 0
                        st.write(f"**{esp}:** Duración esperada: {te_esp:.2f} días, Desviación estándar: {std_esp:.2f} días")
                        objetivo = st.number_input(f"Plazo objetivo para {esp}", min_value=1.0, value=float(te_esp), key=f"obj_{esp}")
                        prob = 0.0
                        if std_esp > 0:
                            z = (objetivo - te_esp) / std_esp
                            prob = float(100 * (0.5 * (1 + math.erf(z/math.sqrt(2)))))
                        st.write(f"Probabilidad de concluir en {objetivo:.1f} días o menos: {prob:.2f}%")
                except Exception as prob_error:
                    st.error(f"Error al calcular probabilidades: {str(prob_error)}")
                
                st.markdown("---")
                st.subheader("5. Cronograma de Ejecución (Gantt por Especialidad)")
                gantt_img_path = None
                try:
                    gantt_data = []
                    start_date = datetime(2023, 10, 1)
                    
                    for code, v in activities.items():
                        if code in cpm_results['es'] and code in pert_results:
                            es = cpm_results['es'][code]
                            duration = pert_results[code]['te']
                            start = start_date + timedelta(days=es)
                            finish = start + timedelta(days=duration)
                            gantt_data.append({
                                'Actividad': f"{code}: {v['name']}",
                                'Inicio': start,
                                'Fin': finish,
                                'Duración': duration,
                                'Especialidad': v['category']
                            })
                    
                    gantt_df = pd.DataFrame(gantt_data)
                    
                    try:
                        import plotly.express as px
                        fig_gantt = px.timeline(gantt_df, x_start='Inicio', x_end='Fin', y='Actividad', 
                                              color='Especialidad', title='Cronograma de Ejecución')
                        fig_gantt.update_layout(
                            height=800,
                            font_size=12,
                            margin=dict(l=50, r=50, b=100, t=100, pad=4),
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        fig_gantt.update_xaxes(
                            tickangle=45,
                            tickformat="%d-%b-%Y",
                            tickfont=dict(size=10)
                        )
                        fig_gantt.update_yaxes(
                            tickfont=dict(size=10),
                            autorange="reversed"
                        )
                        try:
                            st.plotly_chart(fig_gantt, use_container_width=True)
                            gantt_img_path = "diagrama_gantt.png"
                            fig_gantt.write_image(gantt_img_path, width=1200, height=800, scale=2)
                        except Exception as e:
                            if "kaleido" in str(e).lower():
                                st.warning("""
                                **Error al generar gráfico:** Se requiere kaleido para exportar imágenes.
                                Por favor instale kaleido con:
                                pip install kaleido==0.2.1 --no-deps --force-reinstall
                                """)
                            else:
                                st.warning(f"No se pudo mostrar el gráfico Gantt: {str(e)}")
                            st.dataframe(gantt_df)
                            gantt_img_path = None
                    except Exception as gantt_error:
                        st.warning(f"No se pudo generar el gráfico Gantt: {str(gantt_error)}")
                        st.dataframe(gantt_df)
                        gantt_img_path = None
                        
                except Exception as crono_error:
                    st.error(f"Error al generar cronograma: {str(crono_error)}")
                    gantt_img_path = None
                
                st.subheader("6. Cronograma de Adquisición de Materiales y Consolidado Total")
                try:
                    materials_df = get_materials_data()
                    
                    # Asignar fechas de adquisición según inicio de cada actividad
                    for idx, row in materials_df.iterrows():
                        code = row['Actividad']
                        if code in cpm_results['es']:
                            es = cpm_results['es'][code]
                            start = start_date + timedelta(days=int(es))
                            if isinstance(start, datetime):
                                materials_df.at[idx, 'Fecha Necesaria'] = start.strftime('%Y-%m-%d')
                            else:
                                # Si por alguna razón start no es datetime, conviértelo
                                materials_df.at[idx, 'Fecha Necesaria'] = (datetime.now() + timedelta(days=int(es))).strftime('%Y-%m-%d')
                    
                    st.dataframe(materials_df, use_container_width=True)
                except Exception as materials_error:
                    st.error(f"Error al generar cronograma de materiales: {str(materials_error)}")
                
                st.subheader("Exportar resultados a PDF")
                try:
                    if img_path and os.path.exists(img_path) and 'table_df' in locals():
                        pdf_buf = export_to_pdf(img_path, table_df, gantt_img_path)
                        if pdf_buf is not None:
                            st.download_button("Descargar PDF de Diagrama y Tabla", 
                                             data=pdf_buf.getvalue(), 
                                             file_name="PERT_CPM_RESULTADOS.pdf", 
                                             mime="application/pdf")
                            st.markdown('<a href="PERT_CPM_RESULTADOS.pdf" download>Descargar PDF guardado automáticamente</a>', 
                                      unsafe_allow_html=True)
                        else:
                            st.error("No se pudo generar el PDF.")
                    else:
                        st.warning("No se puede exportar PDF sin diagrama o tabla.")
                except Exception as pdf_error:
                    st.error(f"Error al exportar PDF: {str(pdf_error)}")
                
                # Limpiar memoria al final
                gc.collect()
                
            except Exception as e:
                st.error(f"Error en el cálculo: {str(e)}")
                st.info("Intente reducir los valores de entrada o reinicie la aplicación.")
                # Limpiar memoria en caso de error
                gc.collect()
                plt.close('all')
        else:
            st.info("Ingrese los tiempos PERT y presione 'Calcular PERT y CPM' para ver resultados.")
            
    except Exception as main_error:
        st.error(f"Error general en la aplicación: {str(main_error)}")
        st.info("Por favor verifique que todas las dependencias estén instaladas correctamente.")

if __name__ == "__main__":
    main()