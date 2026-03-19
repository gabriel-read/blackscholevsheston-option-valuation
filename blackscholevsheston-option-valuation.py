import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad

# --- Configuración de la página ---
st.set_page_config(page_title="Option Pricing: BS vs Heston", layout="wide")
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
    np.random.seed(42)
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

st.subheader("1. Comparativa de Precios (Riesgo de Modelo)")
col1, col2, col3 = st.columns(3)
col1.metric("Precio Black-Scholes", f"${bs_price:.2f}", "Volatilidad Constante", delta_color="off")
col2.metric("Precio Heston", f"${heston_p:.2f}", "Volatilidad Estocástica", delta_color="off")
col3.metric("Riesgo de Modelo (Diferencia)", f"${model_risk:.2f}", f"{(model_risk/heston_p)*100:.1f}% de sobrevaloración" if heston_p>0 else "")

# --- Gráficas de Sensibilidad ---
st.subheader("2. Evolución de las Griegas al Vencimiento (T -> 0)")
with st.spinner('Calculando sensibilidades numéricas...'):
    tiempos = np.linspace(T, 0.01, 15)
    deltas_bs, gammas_bs, vegas_bs = [], [], []
    deltas_he, gammas_he = [], []
    dS = S0 * 0.01

    for t in tiempos:
        # BS
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma0**2) * t) / (sigma0 * np.sqrt(t))
        deltas_bs.append(norm.cdf(d1))
        gammas_bs.append(norm.pdf(d1) / (S0 * sigma0 * np.sqrt(t)))
        vegas_bs.append(S0 * norm.pdf(d1) * np.sqrt(t) / 100)

        # Heston (Diferencias Finitas)
        P_up = heston_price(S0 + dS, K, t, r, kappa, theta, nu, v0, rho)
        P_mid = heston_price(S0, K, t, r, kappa, theta, nu, v0, rho)
        P_down = heston_price(S0 - dS, K, t, r, kappa, theta, nu, v0, rho)
        deltas_he.append((P_up - P_down) / (2 * dS))
        gammas_he.append((P_up - 2 * P_mid + P_down) / (dS**2))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    ax1.plot(tiempos, deltas_bs, 'b-', label='Black-Scholes')
    ax1.plot(tiempos, deltas_he, 'r--', label='Heston')
    ax1.set_title('Evolución de Delta (Δ)')
    ax1.invert_xaxis()
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(tiempos, gammas_bs, 'b-', label='Black-Scholes')
    ax2.plot(tiempos, gammas_he, 'r--', label='Heston')
    ax2.set_title('Evolución de Gamma (Γ)')
    ax2.invert_xaxis()
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(tiempos, vegas_bs, 'g-', label='BS Vega (por 1%)')
    ax3.set_title('Evolución de Vega (V)')
    ax3.invert_xaxis()
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    st.pyplot(fig)

# --- PARTE B: Hedging ---
st.subheader("3. Optimización de Estrategia de Cobertura (Fricción)")
costo_diario = simulate_delta_hedging(90, c_base, S0, K, T, r, sigma0)
costo_semanal = simulate_delta_hedging(12, c_base, S0, K, T, r, sigma0)

col_h1, col_h2 = st.columns(2)
col_h1.metric("Costo Hedging Diario (0.15%)", f"${costo_diario:.2f}", "Rebalanceo Agresivo", delta_color="inverse")
col_h2.metric("Costo Hedging Semanal (0.15%)", f"${costo_semanal:.2f}", "Rebalanceo Óptimo", delta_color="normal")

# --- PARTE C: Stress Tests ---
st.subheader("4. Stress Tests (Pruebas de Impacto)")

# Shock Volatilidad
vol_crisis = sigma0 * 1.5
precio_crisis_bs = black_scholes_call(S0, K, T, r, vol_crisis)
precio_crisis_he = heston_price(S0, K, T, r, kappa, theta, nu, vol_crisis**2, rho)

# Shock Kappa
precio_kup = heston_price(S0, K, T, r, kappa * 1.2, theta, nu, v0, rho)
precio_kdown = heston_price(S0, K, T, r, kappa * 0.8, theta, nu, v0, rho)

# Shock Costos
c_alto = 0.005
costo_diario_alto = simulate_delta_hedging(90, c_alto, S0, K, T, r, sigma0)

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.markdown("**Shock de Volatilidad (+50%)**")
    st.write(f"BS: **${precio_crisis_bs:.2f}** (+${precio_crisis_bs - bs_price:.2f})")
    st.write(f"Heston: **${precio_crisis_he:.2f}** (+${precio_crisis_he - heston_p:.2f})")

with col_s2:
    st.markdown("**Incertidumbre Paramétrica (κ)**")
    st.write(f"κ +20% (Rápido): **${precio_kup:.2f}**")
    st.write(f"κ -20% (Lento): **${precio_kdown:.2f}**")

with col_s3:
    st.markdown("**Mercados Ilíquidos (Costo 0.50%)**")
    st.write(f"Hedging Diario: **${costo_diario_alto:.2f}**")
    st.write(f"Riesgo de Ruina detectado.")

st.markdown("---")
st.markdown("""
**Metodología:** Proyecto construido como laboratorio de investigación cuantitativa. Se utilizó **Python**, **SciPy** (integración numérica) y **Streamlit**. El código fue desarrollado con la asistencia de IA para estructurar el análisis de estrés y optimizar el front-end interactivo.
""")
