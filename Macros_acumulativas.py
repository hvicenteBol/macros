import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import streamlit as st
from datetime import datetime
from pandas_datareader import data as pdr
import warnings
warnings.filterwarnings('ignore')

# ============== CONFIGURACIÓN STREAMLIT ==============
st.set_page_config(page_title="Presidential Economics Dashboard", layout="wide")
st.title("📊 Presidential Economics Dashboard")

# ============== CONFIGURACIÓN DE VARIABLES MACRO ==============
MACRO_VARIABLES = [
    {
        "name": "S&P500",
        "id": "^GSPC",
        "source": "yahoo",
        "column": "Close",
        "real_adjust": True,
        "cumulative_return": True
    },
    {
        "name": "NASDAQ",
        "id": "^IXIC",
        "source": "yahoo",
        "column": "Close",
        "real_adjust": True,
        "cumulative_return": True
    },
    {
        "name": "Unemployment Rate",
        "id": "UNRATE",
        "source": "fred",
        "column": "UNRATE",
        "real_adjust": False,
        "cumulative_return": False
    },
    {
        "name": "Inflation (CPI YoY)",
        "id": "CPIAUCSL",
        "source": "fred",
        "column": "CPIAUCSL",
        "real_adjust": False,
        "cumulative_return": False
    },
    {
        "name": "Federal Funds Rate",
        "id": "FEDFUNDS",
        "source": "fred",
        "column": "FEDFUNDS",
        "real_adjust": False,
        "cumulative_return": False
    },
    {
        "name": "10-Year Treasury Yield",
        "id": "GS10",
        "source": "fred",
        "column": "GS10",
        "real_adjust": False,
        "cumulative_return": False
    },
    {
        "name": "Personal Consumption Expenditures (PCE)",
        "id": "PCE",
        "source": "fred",
        "column": "PCE",
        "real_adjust": False,
        "cumulative_return": True
    },
    {
        "name": "M2 Money Stock",
        "id": "M2SL",
        "source": "fred",
        "column": "M2SL",
        "real_adjust": False,
        "cumulative_return": True
    },
    {
        "name": "Nonfarm Payrolls",
        "id": "PAYEMS",
        "source": "fred",
        "column": "PAYEMS",
        "real_adjust": False,
        "cumulative_return": False
    }
]

# Colores para los partidos
party_colors = {
    'Republicano': '#E81B23',
    'Demócrata': '#3333FF'
}

# ============== FUNCIONES ==============

@st.cache_data
def download_macro_data(datos_macro):
    """Descarga datos macroeconómicos"""
    try:
        if datos_macro["source"] == "yahoo":
            df = yf.download(datos_macro["id"], period='max')[[datos_macro["column"]]]
        elif datos_macro["source"] == "fred":
            df = pdr.get_data_fred(datos_macro["id"], start='1800-01-01')
        else:
            raise ValueError("Fuente no soportada")
        return df
    except Exception as e:
        st.error(f"Error descargando datos para {datos_macro['name']}: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def download_cpi(start_date):
    """Descarga y procesa datos de CPI"""
    try:
        cpi = pdr.get_data_fred('CPIAUCNS', start=start_date.strftime('%Y-%m-%d'))
        cpi = cpi.resample('D').interpolate(method='linear')
        cpi = cpi / cpi.iloc[-1]  # Normalizar al valor más reciente
        return cpi
    except Exception as e:
        st.error(f"Error descargando CPI: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def get_presidents_data():
    """Obtiene y procesa datos de presidentes"""
    try:
        presidents = pd.read_html("https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States")[0][['Name (birth–death)', 'Term[16]', 'Party[b][17].1']]
        presidents.columns = ['Name', 'Term', 'Party']
        
        # Procesar columna Term
        presidents['Term'] = presidents['Term'].str.replace(r'\[.*?\]', '', regex=True)
        presidents[['Start', 'End']] = presidents['Term'].str.split(" – ", expand=True)
        presidents['Start'] = pd.to_datetime(presidents['Start'], errors='coerce')
        today = pd.Timestamp.today().normalize()
        presidents['End'] = pd.to_datetime(
            presidents['End'].where(presidents['End'] != 'Incumbent', today.strftime('%B %d, %Y')),
            errors='coerce'
        )
        
        # Cambiar etiquetas de partido
        party_map = {
            'Republican': 'Republicano',
            'Democratic': 'Demócrata',
        }
        presidents['Party'] = presidents['Party'].map(party_map)
        presidents['Name'] = presidents['Name'].str.split('(').str[0].str.strip()
        
        return presidents
    except Exception as e:
        st.error(f"Error obteniendo datos de presidentes: {str(e)}")
        return pd.DataFrame()

def get_nominal_returns(start_date, end_date, precios_macro, datos_macro):
    """Calcula retornos nominales"""
    try:
        period_data = precios_macro.loc[start_date:end_date][datos_macro["column"]]
        if period_data.empty:
            return pd.Series(dtype=float)
        if datos_macro["cumulative_return"]:
            return (1 + period_data.pct_change()).cumprod().fillna(1)
        else:
            return period_data -  period_data.iloc[0] +1 ## Normalizar para que cada periodo empiece en 1
    except:
        return pd.Series(dtype=float)

def get_real_returns(start_date, end_date, macro_data, cpi_data, datos_macro):
    """Calcula retornos reales ajustados por inflación"""
    try:
        prices = macro_data.loc[start_date:end_date][datos_macro["column"]]
        cpi = cpi_data.loc[start_date:end_date]['CPIAUCNS']
        df = pd.concat([prices, cpi], axis=1).dropna()
        df.columns = ['Close', 'CPI']
        real_prices = df['Close'] / df['CPI']
        return (1 + real_prices.pct_change()).cumprod().fillna(1)
    except:
        return pd.Series(dtype=float)

def get_party_periods(df_presidentes):
    """Agrupa períodos consecutivos del mismo partido"""
    party_periods = []
    current_party = None
    current_start = None
    current_end = None
    total_days = 0
    
    for _, pres in df_presidentes.iterrows():
        if pres['Party'] != current_party:
            if current_party is not None:
                party_periods.append({
                    'Party': current_party,
                    'Start': current_start,
                    'End': current_end,
                    'Days': total_days
                })
            current_party = pres['Party']
            current_start = pres['Start']
            total_days = 0
        
        current_end = pres['End']
        total_days += (pres['End'] - pres['Start']).days
    
    # Añadir el último período
    if current_party is not None:
        party_periods.append({
            'Party': current_party,
            'Start': current_start,
            'End': current_end,
            'Days': total_days
        })
    
    return party_periods

def create_plot(var, macro_data, cpi_data, historical_presidents, mostrar_por_partido, selected_presidents=None):
    """Crea el gráfico principal"""
    
    # Determinar si necesitamos uno o dos subgráficos
    if var["real_adjust"]:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), facecolor='#1f1f2e', sharex=True)
        axes = [ax1, ax2]
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 6), facecolor='#1f1f2e')
        axes = [ax1]

    # Aplicar estilo a todos los ejes
    for ax in axes:
        ax.set_facecolor('#1f1f2e')

    # Inicializar listas para manejar leyendas
    handles_nom, labels_nom = [], []
    handles_real, labels_real = [], []

    if mostrar_por_partido:
        # Agrupar por períodos de partido
        party_periods = get_party_periods(historical_presidents)
        cumulative_days = 0
        legend_lines_nom = {}
        legend_lines_real = {}
        
        for period in party_periods:
            party = period['Party']
            start = period['Start']
            end = period['End']

            nominal = get_nominal_returns(start, end, macro_data, var)
            days_nom = cumulative_days + (nominal.index - start).days
            
            if party not in legend_lines_nom:
                line_nom, = ax1.plot(days_nom, nominal, color=party_colors.get(party, '#DDA0DD'), 
                                    linewidth=2, alpha=0.8, label=party)
                legend_lines_nom[party] = line_nom
            else:
                ax1.plot(days_nom, nominal, color=party_colors.get(party, '#DDA0DD'), 
                        linewidth=2, alpha=0.8)

            if var["real_adjust"]:
                real = get_real_returns(start, end, macro_data, cpi_data, var)
                days_real = cumulative_days + (real.index - start).days
                
                if party not in legend_lines_real:
                    line_real, = ax2.plot(days_real, real, color=party_colors.get(party, '#DDA0DD'), 
                                        linewidth=2, alpha=0.8, label=party)
                    legend_lines_real[party] = line_real
                else:
                    ax2.plot(days_real, real, color=party_colors.get(party, '#DDA0DD'), 
                            linewidth=2, alpha=0.8)
            
            cumulative_days += period['Days']

        handles_nom = list(legend_lines_nom.values())
        labels_nom = list(legend_lines_nom.keys())
        if var["real_adjust"]:
            handles_real = list(legend_lines_real.values())
            labels_real = list(legend_lines_real.keys())

    else:
        # Mostrar por presidente individual
        if selected_presidents is None:
            selected_presidents = historical_presidents['Name'].tolist()
        
        filtered_presidents = historical_presidents[historical_presidents['Name'].isin(selected_presidents)]
        colors = sns.color_palette("husl", n_colors=len(filtered_presidents))
        
        for idx, pres in filtered_presidents.iterrows():
            name = pres['Name']
            start, end = pres['Start'], pres['End']

            nominal = get_nominal_returns(start, end, macro_data, var)
            days_nom = (nominal.index - start).days

            if not nominal.empty:
                line_nom, = ax1.plot(days_nom, nominal, color=colors[idx % len(colors)], 
                                   linewidth=2, alpha=0.8)
                handles_nom.append(line_nom)
                labels_nom.append(f"{name} ({(end - start).days} días)")

            if var["real_adjust"]:
                real = get_real_returns(start, end, macro_data, cpi_data, var)
                days_real = (real.index - start).days
                if not real.empty:
                    line_real, = ax2.plot(days_real, real, color=colors[idx % len(colors)], 
                                        linewidth=2, alpha=0.8)
                    handles_real.append(line_real)
                    labels_real.append(f"{name} ({(end - start).days} días)")

    # Personalización
    if var["real_adjust"]:
        titles = ['Rendimiento Acumulado Nominal', 'Rendimiento Acumulado Real (ajustado por inflación)']
        ylabels = ['Rendimiento acumulado nominal', 'Rendimiento acumulado real']
        
        for ax, title, ylabel in zip([ax1, ax2], titles, ylabels):
            ax.set_xlabel('Días desde el inicio del mandato', color='white')
            ax.set_ylabel(ylabel, color='white')
            ax.set_title(f'{title} - {var["name"]}', color='white', fontsize=14, pad=10)
            ax.grid(True, alpha=0.2)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')

        ax1.legend(handles_nom, labels_nom, loc='center left', bbox_to_anchor=(1, 0.5),
                   facecolor='#1f1f2e', edgecolor='white', labelcolor='white')
        ax2.legend(handles_real, labels_real, loc='center left', bbox_to_anchor=(1, 0.5),
                   facecolor='#1f1f2e', edgecolor='white', labelcolor='white')
    else:
        ax1.set_xlabel('Días desde el inicio del mandato', color='white')
        ax1.set_ylabel(var["name"], color='white')
        ax1.set_title(f'{var["name"]}', color='white', fontsize=14, pad=10)
        ax1.grid(True, alpha=0.2)
        ax1.tick_params(colors='white')
        for spine in ax1.spines.values():
            spine.set_color('white')
        
        ax1.legend(handles_nom, labels_nom, loc='center left', bbox_to_anchor=(1, 0.5),
                   facecolor='#1f1f2e', edgecolor='white', labelcolor='white')

    plt.tight_layout()
    return fig

# ============== INTERFAZ STREAMLIT ==============

# --- Panel lateral de filtros ---
st.sidebar.markdown("## ⚙️ Filtros de Análisis")

# Selección de variable macro
macro_names = [datos_variablemacro["name"] for datos_variablemacro in MACRO_VARIABLES]
indice_macro_seleccionada = st.sidebar.selectbox("📈 Selecciona variable macroeconómica:",  range(len(macro_names)), format_func=lambda i: macro_names[i])

# Cargar datos macro con antelacion para saber de cuantos datos disponemos
with st.spinner(f"Descargando datos de {indice_macro_seleccionada}..."):
    macro_data = download_macro_data(MACRO_VARIABLES[indice_macro_seleccionada])

if macro_data.empty:
    st.error("No se pudieron cargar los datos macroeconómicos")
    st.stop()

# Filtrar presidentes históricos
start_date = macro_data.index[0].date()

# Tipo de visualización
mostrar_por_partido = st.sidebar.radio(
    "👥 Tipo de visualización:",
    ["Por Partido", "Por Presidente"],
    index=0
) == "Por Partido"


# Cargar datos de presidentes
with st.spinner("Cargando datos de presidentes..."):
    presidents_df = get_presidents_data()
    
if presidents_df.empty:
    st.error("No se pudieron cargar los datos de presidentes")
    st.stop()


# Selector de presidentes (solo si no es por partido)
selected_presidents = None
if not mostrar_por_partido:
    # Filtrar presidentes con datos disponibles
    start_date = macro_data.index[0].date()  # Fecha del primer dato macro para seleccionar los presidentes disponibles
    historical_presidents = presidents_df[presidents_df['Start'].dt.year >= start_date.year].reset_index(drop=True)
    
    president_names = [
    f"{row['Name']} ({row['Start'].year if pd.notna(row['Start']) else '???'} - {row['End'].year if pd.notna(row['End']) else '???'})"
    for _, row in historical_presidents.iterrows()
]

    st.sidebar.markdown("### 🎯 Selecciona presidentes:")

    # Estado de selección (usamos session_state para persistencia)
    if "selected_presidents" not in st.session_state:
        st.session_state.selected_presidents = set()

    # Botón para seleccionar todos

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("✅ Seleccionar todos"):
            st.session_state.selected_presidents = set(president_names)

    with col2:
        if st.button("❌ Quitar todos"):
            st.session_state.selected_presidents = set()

    # Expander con checkboxes por presidente
    with st.sidebar.expander("✔️ Presidentes disponibles", expanded=True):
        updated_selection = set()
        for i, name in enumerate(president_names):
            key = f"president_checkbox_{i}"
            checked = st.checkbox(name, value=name in st.session_state.selected_presidents, key=key)
            if checked:
                updated_selection.add(name)
        st.session_state.selected_presidents = updated_selection
        
    selected_presidents = list(st.session_state.selected_presidents)

    if not selected_presidents:
        st.warning("Por favor selecciona al menos un presidente")
        st.stop()


show = st.sidebar.button("📊 Generar Gráfico", type="primary")

# Encontrar la variable seleccionada
datos_variablemacro = next((v for v in MACRO_VARIABLES if v["name"] == MACRO_VARIABLES[indice_macro_seleccionada]["name"]), None)

if datos_variablemacro is None:
    st.error("Variable macroeconómica no encontrada")
    st.stop()


# Botón para generar gráfico
if show:
     
    # Cargar CPI si es necesario
    cpi_data = None
    if datos_variablemacro["real_adjust"]:
        with st.spinner("Descargando datos de inflación..."):
            start_date = macro_data.index[0].date()
            cpi_data = download_cpi(start_date)
        
    # Crear gráfico
    with st.spinner("Generando gráfico..."):
        fig = create_plot(datos_variablemacro, macro_data, cpi_data, historical_presidents, mostrar_por_partido, selected_presidents)
    
    # Mostrar gráfico
    st.pyplot(fig)
    
    # Información adicional
    st.subheader("ℹ️ Información sobre la variable")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Variable", datos_variablemacro["name"])
    with col2:
        st.metric("Fuente", datos_variablemacro["source"].upper())
    with col3:
        st.metric("Ajuste inflación", "Sí" if datos_variablemacro["real_adjust"] else "No")
    
    if not mostrar_por_partido and selected_presidents:
        st.subheader("👥 Presidentes seleccionados")
        filtered_presidents = historical_presidents[historical_presidents['Name'].isin(selected_presidents)]
        
        # Mostrar tabla con información de presidentes
        display_df = filtered_presidents[['Name', 'Party', 'Start', 'End']].copy()
        display_df['Duración (días)'] = (display_df['End'] - display_df['Start']).dt.days
        display_df['Start'] = display_df['Start'].dt.strftime('%Y-%m-%d')
        display_df['End'] = display_df['End'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(display_df, use_container_width=True)

else:
    st.info("👆 Configura los parámetros en la barra lateral y haz clic en 'Generar Gráfico' para ver el análisis")
    
    # Mostrar información sobre las variables disponibles
    st.subheader("📊 Variables Macroeconómicas Disponibles")
    
    info_df = pd.DataFrame(MACRO_VARIABLES)
    display_info = info_df[['name', 'source', 'real_adjust', 'cumulative_return']].copy()
    display_info.columns = ['Variable', 'Fuente', 'Ajuste Inflación', 'Retorno Acumulativo']
    
    st.dataframe(display_info, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*Dashboard creado para análisis de variables macroeconómicas durante mandatos presidenciales de EE.UU.*")
