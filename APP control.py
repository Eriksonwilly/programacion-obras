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

def draw_pert_cpm_diagram(activities, cpm_results, pert_results, show_table=False, fig_width=16.5, fig_height=11.7, node_radius=18):
    try:
        gc.collect()
        plt.close('all')
        G = nx.DiGraph()
        for activity in activities:
            G.add_node(activity)
        for activity in activities:
            for pred in activities[activity]['predecessors']:
                G.add_edge(pred, activity)

        # Intentar usar layout jerárquico (graphviz_layout)
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except Exception:
            try:
                pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
            except Exception:
                pos = nx.circular_layout(G)

        # Ajustar tamaño a A3 (11.7 x 16.5 pulgadas)
        fig, ax = plt.subplots(figsize=(16.5, 11.7), facecolor='white')
        plt.rcParams['font.size'] = 9
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['lines.linewidth'] = 2

        for edge in G.edges():
            is_critical = edge[0] in cpm_results['critical_path'] and edge[1] in cpm_results['critical_path']
            color = '#FF0000' if is_critical else '#333333'
            width = 3 if is_critical else 1.5
            alpha = 0.9 if is_critical else 0.7
            ax.annotate('', xy=pos[edge[1]], xytext=pos[edge[0]],
                        arrowprops=dict(arrowstyle="->", color=color, lw=width, 
                                      shrinkA=node_radius+3, shrinkB=node_radius+3,
                                      alpha=alpha, mutation_scale=20))

        for node in G.nodes():
            x, y = pos[node]
            es = cpm_results['es'][node]
            ef = cpm_results['ef'][node]
            ls = cpm_results['ls'][node]
            lf = cpm_results['lf'][node]
            slack = cpm_results['slack'][node]
            te = pert_results[node]['te'] if pert_results else activities[node]['duration']
            is_critical = node in cpm_results['critical_path']
            if is_critical:
                node_color = '#FFE6E6'
                edge_color = '#FF0000'
                edge_width = 2.5
            else:
                node_color = '#E6F3FF'
                edge_color = '#0066CC'
                edge_width = 1.5
            circ = Circle((x, y), node_radius, fill=True, lw=edge_width, 
                         color=node_color, ec=edge_color, alpha=0.95)
            ax.add_patch(circ)
            if is_critical:
                circ2 = Circle((x, y), node_radius+3, fill=False, lw=3, color='#FF0000')
                ax.add_patch(circ2)
            ax.text(x, y+node_radius*0.5, f"{node}", fontsize=12, ha='center', va='center', 
                   fontweight='bold', color='#003366',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor=edge_color))
            activity_name = activities[node]['name']
            if len(activity_name) > 18:
                words = activity_name.split()
                if len(words) > 2:
                    activity_name = ' '.join(words[:2]) + '\n' + ' '.join(words[2:])
                else:
                    activity_name = activity_name[:15] + "..."
            ax.text(x, y, activity_name, fontsize=8, ha='center', va='center', 
                   color='#000000', fontweight='normal', 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
            ax.text(x, y-node_radius*0.5, f"TE={te:.1f}d", fontsize=10, ha='center', va='center', 
                   color='#066666', fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor='#E8F5E8', alpha=0.9))
            ax.text(x-node_radius*0.7, y+node_radius*0.2, f"ES={es:.0f}", fontsize=7, 
                   ha='center', va='center', color='#066666', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.1", facecolor='#E8F5E8', alpha=0.8))
            ax.text(x+node_radius*0.7, y+node_radius*0.2, f"EF={ef:.0f}", fontsize=7, 
                   ha='center', va='center', color='#066666', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.1", facecolor='#E8F5E8', alpha=0.8))
            ax.text(x-node_radius*0.7, y-node_radius*0.2, f"LS={ls:.0f}", fontsize=7, 
                   ha='center', va='center', color='#CC6600', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.1", facecolor='#FFF2E6', alpha=0.8))
            ax.text(x+node_radius*0.7, y-node_radius*0.2, f"LF={lf:.0f}", fontsize=7, 
                   ha='center', va='center', color='#CC6600', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.1", facecolor='#FFF2E6', alpha=0.8))
            if slack > 0.01:
                ax.text(x, y-node_radius*0.8, f"H:{slack:.0f}", fontsize=8, ha='center', va='center', 
                       color='#660066', fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='#F0E6F0', alpha=0.9))
        ax.text(0.5, 1.08, "DIAGRAMA DE RED PERT-CPM", fontsize=18, ha='center', va='center', transform=ax.transAxes, fontweight='bold', color='#003366',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='#E6F3FF', alpha=0.9))
        ax.text(0.5, 1.04, "Proyecto: Construcción de Vivienda de Dos Plantas + Azotea", fontsize=14, ha='center', va='center', transform=ax.transAxes, color='#033666', fontweight='bold')
        ax.text(0.5, 1.01, "Ubicación: Chiclayo, Lambayeque | Empresa: CONSORCIO DEJ", fontsize=11, ha='center', va='center', transform=ax.transAxes, color='#666666')
        legend_x = 1.02
        legend_y = 0.95
        ax.text(legend_x, legend_y, "LEYENDA DEL DIAGRAMA", fontsize=12, ha='left', va='center', transform=ax.transAxes, fontweight='bold', color='#003366',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='#F5F5F5', alpha=0.9))
        ax.text(legend_x, legend_y-0.04, "● Ruta Crítica (Rojo)", fontsize=10, ha='left', va='center', transform=ax.transAxes, color='#FF0000', fontweight='bold')
        ax.text(legend_x, legend_y-0.08, "● Actividad Normal (Azul)", fontsize=10, ha='left', va='center', transform=ax.transAxes, color='#0066CC')
        ax.text(legend_x, legend_y-0.12, "ES: Early Start (Verde)", fontsize=9, ha='left', va='center', transform=ax.transAxes, color='#066666', fontweight='bold')
        ax.text(legend_x, legend_y-0.16, "EF: Early Finish (Verde)", fontsize=9, ha='left', va='center', transform=ax.transAxes, color='#066666', fontweight='bold')
        ax.text(legend_x, legend_y-0.20, "LS: Late Start (Naranja)", fontsize=9, ha='left', va='center', transform=ax.transAxes, color='#CC6600', fontweight='bold')
        ax.text(legend_x, legend_y-0.24, "LF: Late Finish (Naranja)", fontsize=9, ha='left', va='center', transform=ax.transAxes, color='#CC6600', fontweight='bold')
        ax.text(legend_x, legend_y-0.28, "TE: Tiempo Esperado (Verde)", fontsize=9, ha='left', va='center', transform=ax.transAxes, color='#066666', fontweight='bold')
        ax.text(legend_x, legend_y-0.32, "H: Holgura (Púrpura)", fontsize=9, ha='left', va='center', transform=ax.transAxes, color='#660066', fontweight='bold')
        ax.axis('off')
        plt.subplots_adjust(left=0.02, right=0.88, top=0.95, bottom=0.2)
        return fig
    except Exception as e:
        st.error(f"Error al generar diagrama: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return None

def export_to_pdf(fig, table_df):
    try:
        gc.collect()
        buf = io.BytesIO()
        fig.set_size_inches(23.4, 16.5)
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        buf.seek(0)
        pdf_buf = io.BytesIO()
        c = canvas.Canvas(pdf_buf, pagesize=landscape(A2))
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, 550, "REPORTE PERT-CPM - CONTROL DE OBRAS")
        c.setFont("Helvetica", 12)
        c.drawString(40, 530, "Proyecto: Construcción de Vivienda de Dos Plantas + Azotea")
        c.drawString(40, 515, "Ubicación: Chiclayo, Lambayeque | Empresa: CONSORCIO DEJ")
        try:
            img = ImageReader(buf)
            c.drawImage(img, 40, 250, width=900, height=500, mask='auto')
        except Exception as e:
            c.drawString(40, 300, f"Error al insertar diagrama: {str(e)}")
        x0, y0 = 40, 220
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x0, y0, "Tabla de Actividades y Resultados PERT:")
        y = y0 - 20
        col_widths = [50, 180, 70, 50, 50, 50, 50, 80]
        headers = list(table_df.columns)
        c.setFont("Helvetica-Bold", 9)
        for i, h in enumerate(headers):
            c.drawString(x0 + sum(col_widths[:i]), y, str(h))
        y -= 15
        c.setFont("Helvetica", 8)
        for idx, row in table_df.iterrows():
            for i, val in enumerate(row):
                text_val = str(round(val,2)) if isinstance(val, float) else str(val)
                c.drawString(x0 + sum(col_widths[:i]), y, text_val)
            y -= 12
            if y < 30:
                c.showPage()
                y = 500
                c.setFont("Helvetica", 8)
        c.showPage()
        c.save()
        pdf_buf.seek(0)
        # Guardar automáticamente, pero manejar errores y no detener la app
        try:
            with open("PERT_CPM_RESULTADOS.pdf", "wb") as f:
                f.write(pdf_buf.getvalue())
        except Exception as e:
            st.warning(f"No se pudo guardar el archivo PDF automáticamente en disco: {str(e)}. Puedes descargarlo usando el botón de descarga.")
        return pdf_buf
    except Exception as e:
        st.error(f"Error al exportar PDF: {str(e)}")
        return None

def main():
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
            
            st.subheader("2. Diagrama de Red PERT-CPM (con tiempos esperados y especialidad)")
            
            # Generar diagrama con parámetros optimizados
            try:
                fig = draw_pert_cpm_diagram(activities, cpm_results, pert_results, 
                                          fig_width=16.5, fig_height=11.7, node_radius=18)
                
                if fig is not None:
                    st.pyplot(fig, use_container_width=False)
                    plt.close(fig)  # Cerrar figura para liberar memoria
                else:
                    st.warning("No se pudo generar el diagrama debido a limitaciones de memoria. Pruebe reducir el tamaño del diagrama o reinicie la app.")
            except Exception as diagram_error:
                st.error(f"Error al generar diagrama: {str(diagram_error)}")
                st.info("Sugerencia: Si el error es 'bad allocation', pruebe reducir el tamaño del diagrama o reinicie la app.")
            
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
                if fig is not None and 'table_df' in locals():
                    pdf_buf = export_to_pdf(fig, table_df)
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

if __name__ == "__main__":
    main()