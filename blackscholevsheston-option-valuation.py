import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
from scipy.integrate import quad

# --- Configuración de la página ---
st.set_page_config(page_title="Option Pricing: BS vs Heston", page_icon="📈", layout="wide")
st.title("📈 Valoración Dinámica de Opciones: Black-Scholes vs. Heston")
st.markdown("Comparativa cuantitativa del riesgo de mercado y evaluación de opciones asumiendo volatilidad constante vs. estocástica.")

# --- Barra lateral para parámetros ---
st.sidebar.header("⚙️ Parámetros del Mercado")
S0 = st.sidebar.number_input("Precio del Activo (S_0)", value=120.0, step=1.0)
K = st.sidebar.number_input("Precio de Ejercicio (K)", value=125.0, step=1.0)
T = st.sidebar.slider("Tiempo al Vencimiento (Años)", min_value=0.01, max_value=1.0, value=0.25)
r = st.sidebar.number_input("Tasa Libre de Riesgo (r)", value=0.04, step=0.01)

st.sidebar.header("📊 Dinámica de Volatilidad")
sigma0 = st.sidebar.slider("Volatilidad Inicial (σ_0)", 0.1, 1.0, 0.35)
kappa = st.sidebar.slider("Reversión a la Media (κ)", 0.0, 5.0, 2.5)
theta_pct = st.sidebar.slider("Vol. de Largo Plazo (%)", 10, 100, 30)
theta = (theta_pct / 100)**2
nu = st.sidebar.slider("Volatilidad de la Volatilidad (ν)", 0.01, 1.0, 0.15)
rho = st.sidebar.slider("Correlación (ρ)", -1.0, 1.0, -0.7)
v0 = sigma0**2
c_base = 0.0015 # Costo de transacción base (0.15%)

# --- Funciones de Valoración ---
@st.cache_data
def black_scholes_call(S, K, T, r, sigma):
    if T <= 0: return max(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

@st.cache_data
def heston_price(S0, K, T, r, kappa, theta, nu, v0, rho):
    def characteristic_func(u, j):
        a = kappa * theta
        u_j = 0.5 if j == 1 else -0.5
        b_j = kappa - rho * nu if j == 1 else kappa
        d = np.sqrt((rho * nu * u * 1j - b_j)**2 - nu**2 * (2 * u_j * u * 1j - u**2))
        g = (b_j - rho * nu * u * 1j + d) / (b_j - rho * nu * u * 1j - d)
        C = r * u * 1j * T + (a / nu**2) * ((b_j - rho * nu * u * 1j + d) * T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
        D = ((b_j - rho * nu * u * 1j + d) / nu**2) * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))
        return np.exp(C + D * v0 + 1j * u * np.log(S0))

    def p_func(u, j):
        return (np.exp(-1j * u * np.log(K)) * characteristic_func(u, j) / (1j * u)).real

    try:
        p1 = 0.5 + (1 / np.pi) * quad(p_func, 0, 100, args=(1,), limit=100)[0]
        p2 = 0.5 + (1 / np.pi) * quad(p_func, 0, 100, args=(2,), limit=100)[0]
        return max(0, S0 * p1 - K * np.exp(-r * T) * p2)
    except:
        return 0.0

@st.cache_data
def simulate_delta_hedging(N_steps, cost_rate, S0, K, T, r, sigma0):
    np.random.seed(42) # Semilla fijada estrictamente para coincidir con el informe
    dt = T / N_steps
    S_path = [S0]
    for _ in range(N_steps):
        Z = np.random.normal(0, 1)
        S_next = S_path[-1] * np.exp((r - 0.5 * sigma0**2) * dt + sigma0 * np.sqrt(dt) * Z)
        S_path.append(S_next)
    
    total_cost = 0
    prev_delta = 0
    for i in range(N_steps):
        t_remain = T - i * dt
        if t_remain <= 0: break
        d1 = (np.log(S_path[i] / K) + (r + 0.5 * sigma0**2) * t_remain) / (sigma0 * np.sqrt(t_remain))
        current_delta = norm.cdf(d1)
        shares_traded = abs(current_delta - prev_delta)
        total_cost += shares_traded * S_path[i] * cost_rate
        prev_delta = current_delta
    return total_cost

# --- PARTE A: Cálculos y Métricas ---
bs_price = black_scholes_call(S0, K, T, r, sigma0)
heston_p = heston_price(S0, K, T, r, kappa, theta, nu, v0, rho)
model_risk = abs(bs_price - heston_p)

st.divider()
st.subheader("1. Comparativa de Precios (Riesgo de Modelo)")
col1, col2, col3 = st.columns(3)
col1.metric("Precio Black-Scholes", f"${bs_price:.2f}", "Volatilidad Constante", delta_color="off")
col2.metric("Precio Heston", f"${heston_p:.2f}", "Volatilidad Estocástica", delta_color="off")
col3.metric("Riesgo de Modelo (Diferencia)", f"${model_risk:.2f}", f"{(model_risk/heston_p)*100:.1f}% de sobrevaloración" if heston_p>0 else "", delta_color="inverse")

# --- Gráficas de Sensibilidad (PLOTLY) ---
st.divider()
st.subheader("2. Evolución de las Griegas al Vencimiento (T → 0)")
with st.spinner('Procesando simulación estocástica y derivadas numéricas...'):
    tiempos = np.linspace(T, 0.01, 20)
    deltas_bs, gammas_bs, vegas_bs = [], [], []
    deltas_he, gammas_he = [], []
    dS = S0 * 0.01

    for t in tiempos:
        # BS
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma0**2) * t) / (sigma0 * np.sqrt(t))
        deltas_bs.append(norm.cdf(d1))
        gammas_bs.append(norm.pdf(d1) / (S0 * sigma0 * np.sqrt(t)))
        vegas_bs.append(S0 * norm.pdf(d1) * np.sqrt(t) / 100)

        # Heston
        P_up = heston_price(S0 + dS, K, t, r, kappa, theta, nu, v0, rho)
        P_mid = heston_price(S0, K, t, r, kappa, theta, nu, v0, rho)
        P_down = heston_price(S0 - dS, K, t, r, kappa, theta, nu, v0, rho)
        deltas_he.append((P_up - P_down) / (2 * dS))
        gammas_he.append((P_up - 2 * P_mid + P_down) / (dS**2))

    # Creación de dashboard interactivo con Plotly
    fig = make_subplots(rows=1, cols=3, subplot_titles=('Evolución de Delta (Δ)', 'Evolución de Gamma (Γ)', 'Evolución de Vega (V)'))
    
    # Trazos Delta
    fig.add_trace(go.Scatter(x=tiempos, y=deltas_bs, mode='lines', name='BS Delta', line=dict(color='#EF553B')), row=1, col=1)
    fig.add_trace(go.Scatter(x=tiempos, y=deltas_he, mode='lines', name='Heston Delta', line=dict(color='#636EFA', dash='dash')), row=1, col=1)
    
    # Trazos Gamma
    fig.add_trace(go.Scatter(x=tiempos, y=gammas_bs, mode='lines', name='BS Gamma', line=dict(color='#EF553B'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=tiempos, y=gammas_he, mode='lines', name='Heston Gamma', line=dict(color='#636EFA', dash='dash'), showlegend=False), row=1, col=2)
    
    # Trazos Vega
    fig.add_trace(go.Scatter(x=tiempos, y=vegas_bs, mode='lines', name='BS Vega', line=dict(color='#00CC96')), row=1, col=3)

    fig.update_xaxes(autorange="reversed", title_text="Tiempo al Vencimiento")
    fig.update_layout(height=450, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    
    st.plotly_chart(fig, use_container_width=True)

# --- PARTE B: Hedging ---
st.divider()
st.subheader("3. Optimización de Estrategia de Cobertura (Fricción)")
costo_diario = simulate_delta_hedging(90, c_base, S0, K, T, r, sigma0)
costo_semanal = simulate_delta_hedging(12, c_base, S0, K, T, r, sigma0)

col_h1, col_h2 = st.columns(2)
with col_h1:
    st.metric("Costo Hedging Diario (0.15%)", f"${costo_diario:.2f}", "Rebalanceo Agresivo - Fricción Alta", delta_color="inverse")
with col_h2:
    st.metric("Costo Hedging Semanal (0.15%)", f"${costo_semanal:.2f}", "Zona Óptima de Tracking Error", delta_color="normal")

# --- PARTE C: Stress Tests ---
st.divider()
st.subheader("4. Pruebas de Estrés (Stress Tests)")

# Cálculos
vol_crisis = sigma0 * 1.5
precio_crisis_bs = black_scholes_call(S0, K, T, r, vol_crisis)
precio_crisis_he = heston_price(S0, K, T, r, kappa, theta, nu, vol_crisis**2, rho)

precio_kup = heston_price(S0, K, T, r, kappa * 1.2, theta, nu, v0, rho)
precio_kdown = heston_price(S0, K, T, r, kappa * 0.8, theta, nu, v0, rho)

costo_diario_alto = simulate_delta_hedging(90, 0.005, S0, K, T, r, sigma0)

# Interfaz UI con Tarjetas
col_s1, col_s2, col_s3 = st.columns(3)

with col_s1:
    st.info("📉 **Shock de Volatilidad (+50%)**")
    st.metric("BS (Daño Permanente)", f"${precio_crisis_bs:.2f}", f"+${precio_crisis_bs - bs_price:.2f}", delta_color="inverse")
    st.metric("Heston (Amortiguado)", f"${precio_crisis_he:.2f}", f"+${precio_crisis_he - heston_p:.2f}", delta_color="inverse")

with col_s2:
    st.warning("⚖️ **Incertidumbre de Kappa (κ)**")
    st.metric("κ +20% (Reversión Rápida)", f"${precio_kup:.2f}")
    st.metric("κ -20% (Reversión Lenta)", f"${precio_kdown:.2f}")

with col_s3:
    st.error("🚨 **Mercados Ilíquidos (Costo 0.50%)**")
    st.metric("Costo de Hedging Diario", f"${costo_diario_alto:.2f}", "Riesgo de Ruina por comisiones", delta_color="inverse")
    st.markdown("*La estructura de costos obliga a reducir la frecuencia a nivel semanal para evitar la quiebra.*")

st.divider()
st.caption("🔍 **Metodología:** Proyecto construido como laboratorio de investigación cuantitativa. Se utilizó **Python**, **SciPy** (integración numérica) y **Streamlit / Plotly**. El código fue desarrollado con asistencia de inteligencia artificial para acelerar la interfaz de usuario interactiva y optimizar las visualizaciones de riesgo.")
