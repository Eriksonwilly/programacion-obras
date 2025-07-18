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
import importlib.util

# Verificar si kaleido está disponible para exportar imágenes de Plotly
try:
    import kaleido  # Necesario para exportar imágenes de Plotly
    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False

def check_dependencies():
    """Verifica si plotly-orca está instalado"""
    try:
        import plotly.io as pio
        if 'orca' in pio.orca.status:
            return True
        return False
    except:
        return False

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
        nx.draw_networkx_nodes(G, pos, node_size=node_size, 
                              node_color=node_colors, 
                              edgecolors=edge_colors,
                              linewidths=1.5,
                              ax=ax)
        
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
        
        # Sección 1: Diagrama de Red
        current_y = top_margin - 70
        c.setFont("Helvetica-Bold", 14)
        c.drawString(left_margin, current_y, "1. Diagrama de Red PERT-CPM")
        current_y -= 20
        
        try:
            if img_path and os.path.exists(img_path):
                img = ImageReader(img_path)
                # Ajustar tamaño para mejor visualización en PDF
                img_width = 900
                img_height = 500
                c.drawImage(img, left_margin, current_y - img_height, 
                          width=img_width, height=img_height, mask='auto')
                current_y -= img_height + section_spacing
            else:
                c.drawString(left_margin, current_y, "No se pudo cargar el diagrama de red.")
                current_y -= section_spacing
        except Exception as e:
            c.drawString(left_margin, current_y, f"Error al insertar diagrama de red: {str(e)}")
            current_y -= section_spacing
        
        # Sección 2: Diagrama de Gantt
        if gantt_img_path and os.path.exists(gantt_img_path):
            c.setFont("Helvetica-Bold", 14)
            c.drawString(left_margin, current_y, "2. Diagrama de Gantt")
            current_y -= 20
            
            try:
                gantt_img = ImageReader(gantt_img_path)
                gantt_width = 900
                gantt_height = 400
                c.drawImage(gantt_img, left_margin, current_y - gantt_height, 
                          width=gantt_width, height=gantt_height, mask='auto')
                current_y -= gantt_height + section_spacing
            except Exception as e:
                c.drawString(left_margin, current_y, f"Error al insertar diagrama Gantt: {str(e)}")
                current_y -= section_spacing
        
        # Sección 3: Tabla de actividades
        c.setFont("Helvetica-Bold", 14)
        c.drawString(left_margin, current_y, "3. Tabla de Actividades y Resultados PERT")
        current_y -= 20
        
        # Configurar tabla
        col_widths = [50, 180, 70, 50, 50, 50, 50, 80]  # Ajustar anchos de columnas
        row_height = 15
        headers = list(table_df.columns)
        
        # Dibujar encabezados de tabla
        c.setFont("Helvetica-Bold", 9)
        for i, h in enumerate(headers):
            c.drawString(left_margin + sum(col_widths[:i]), current_y, str(h))
        current_y -= row_height
        
        # Dibujar filas de la tabla
        c.setFont("Helvetica", 8)
        for idx, row in table_df.iterrows():
            # Verificar si necesitamos nueva página
            if current_y < 100:
                c.showPage()
                current_y = top_margin - 40
                # Redibujar encabezados si hay cambio de página
                c.setFont("Helvetica-Bold", 9)
                for i, h in enumerate(headers):
                    c.drawString(left_margin + sum(col_widths[:i]), current_y, str(h))
                current_y -= row_height
                c.setFont("Helvetica", 8)
            
            for i, val in enumerate(row):
                text_val = str(round(val,2)) if isinstance(val, float) else str(val)
                c.drawString(left_margin + sum(col_widths[:i]), current_y, text_val)
            current_y -= row_height
        
        c.showPage()
        c.save()
        pdf_buf.seek(0)
        
        # Guardar automáticamente
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
        # Verificar dependencias críticas
        try:
            import streamlit as st
            import pandas as pd
            import networkx as nx
            import matplotlib.pyplot as plt
            import numpy as np
            import math
            from datetime import datetime, timedelta
        except ImportError as import_error:
            st.error(f"Error de importación: {str(import_error)}. Por favor instale las dependencias con: pip install -r requirements.txt")
            return
        
        # Verificar y mostrar estado de dependencias para exportación
        ORCA_AVAILABLE = check_dependencies()
        if not KALEIDO_AVAILABLE and not ORCA_AVAILABLE:
            st.warning("""
                **Nota:** Para una mejor exportación de imágenes, instale alguno de estos paquetes:
                - `pip install kaleido` (recomendado)
                - `pip install plotly-orca`
                
                Sin estos, la exportación del diagrama Gantt podría no funcionar correctamente.
            """)
        
        activities = get_project_data()
        st.title("Presentación Integral PERT-CPM y Control de Obras")
        st.markdown("""
        <b>Proyecto:</b> Construcción de Vivienda de Dos Plantas + Azotea  
        <b>Ubicación:</b> Chiclayo, Lambayeque  
        <b>Elaborado por:</b> Ing. Civil UNI (20 años de experiencia en edificación)  
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
                        fig_gantt.update_layout(height=600)
                        st.plotly_chart(fig_gantt, use_container_width=True)
                    except Exception as gantt_error:
                        st.warning(f"No se pudo generar el gráfico Gantt: {str(gantt_error)}")
                        st.dataframe(gantt_df)
                except Exception as crono_error:
                    st.error(f"Error al generar cronograma: {str(crono_error)}")
                
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
                    # Generar y guardar diagrama Gantt como imagen
                    gantt_img_path = None
                    if 'fig_gantt' in locals():
                        try:
                            gantt_img_path = "diagrama_gantt.png"
                            
                            # Intentar con kaleido primero
                            if KALEIDO_AVAILABLE:
                                try:
                                    fig_gantt.write_image(gantt_img_path, engine='kaleido', width=1000, height=600, scale=2)
                                except Exception as kaleido_error:
                                    st.warning("Kaleido está instalado pero no pudo exportar la imagen. Intentando alternativas...")
                                    # Intentar con orca si está disponible
                                    if ORCA_AVAILABLE:
                                        try:
                                            fig_gantt.write_image(gantt_img_path, engine='orca', width=1000, height=600, scale=2)
                                        except:
                                            # Último intento sin engine específico
                                            fig_gantt.write_image(gantt_img_path, width=1000, height=600, scale=2)
                                    else:
                                        fig_gantt.write_image(gantt_img_path, width=1000, height=600, scale=2)
                            else:
                                # Si kaleido no está disponible, intentar con plotly directamente
                                fig_gantt.write_image(gantt_img_path, width=1000, height=600, scale=2)
                                
                        except Exception as gantt_img_error:
                            st.warning("""
                                No se pudo guardar el diagrama Gantt como imagen usando Plotly. 
                                Soluciones posibles:
                                1. Instale Google Chrome en su sistema
                                2. O instale Kaleido con: `pip install kaleido`
                                3. O instale Orca con: `pip install plotly-orca`
                            """)
                            
                            # Alternativa: Guardar como imagen usando matplotlib
                            try:
                                st.warning("Intentando exportar con matplotlib como alternativa...")
                                plt.figure(figsize=(12, 6))
                                for esp in gantt_df['Especialidad'].unique():
                                    df_esp = gantt_df[gantt_df['Especialidad'] == esp]
                                    plt.barh(df_esp['Actividad'], 
                                            (df_esp['Fin'] - df_esp['Inicio']).dt.days, 
                                            left=df_esp['Inicio'], 
                                            label=esp)
                                
                                plt.xlabel('Fecha')
                                plt.ylabel('Actividad')
                                plt.title('Diagrama de Gantt')
                                plt.legend()
                                plt.tight_layout()
                                gantt_img_path = "diagrama_gantt_matplotlib.png"
                                plt.savefig(gantt_img_path, bbox_inches='tight', dpi=120)
                                plt.close()
                                st.success("Se generó el diagrama Gantt con matplotlib como alternativa")
                                
                            except Exception as matplotlib_error:
                                st.warning(f"No se pudo generar el diagrama Gantt con matplotlib: {str(matplotlib_error)}")
                                # Último recurso: guardar como HTML
                                try:
                                    gantt_img_path = "diagrama_gantt.html"
                                    fig_gantt.write_html(gantt_img_path)
                                    st.warning("Se guardó el diagrama Gantt como HTML temporal. No se incluirá en el PDF.")
                                    gantt_img_path = None
                                except Exception as html_error:
                                    st.warning(f"No se pudo guardar el diagrama Gantt en ningún formato: {str(html_error)}")
                    
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