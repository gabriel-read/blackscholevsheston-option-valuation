import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad

# ==========================================
# --- Parámetros del Sistema ---
# ==========================================
S0 = 120.0      # Precio actual de la acción 
K = 125.0       # Precio de ejercicio 
T = 0.25        # Tiempo (90 días) 
r = 0.04        # Tasa libre de riesgo 
sigma0 = 0.35   # Volatilidad inicial 
kappa = 2.5     # Velocidad de reversión 
theta = 0.30**2 # Varianza de largo plazo (30%^2) 
v0 = sigma0**2  # Varianza inicial 
nu = 0.15       # Volatilidad de la volatilidad 
rho = -0.7      # Correlación (típica en tecnología)
c_base = 0.0015 # Costo de transacción base (0.15%) 

# ==========================================
# --- 1. Modelos de Valoración ---
# ==========================================
def black_scholes_call(S, K, T, r, sigma):
    if T <= 0: return max(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

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

    p1 = 0.5 + (1 / np.pi) * quad(p_func, 0, 100, args=(1,), limit=100)[0]
    p2 = 0.5 + (1 / np.pi) * quad(p_func, 0, 100, args=(2,), limit=100)[0]
    
    return S0 * p1 - K * np.exp(-r * T) * p2

# ==========================================
# --- PARTE A: Comparación y Griegas ---
# ==========================================
print("=== PARTE A: Valoración de Modelos ===")
bs_p = black_scholes_call(S0, K, T, r, sigma0)
he_p = heston_price(S0, K, T, r, kappa, theta, nu, v0, rho)

print(f"Precio Black-Scholes: ${bs_p:.2f}")
print(f"Precio Heston: ${he_p:.2f}")
print(f"Riesgo de Modelo (Diferencia): ${abs(bs_p - he_p):.2f}\n")

# Cálculo de Griegas a lo largo del tiempo
tiempos = np.linspace(0.25, 0.01, 20) # Reducido a 20 pasos para optimizar rendimiento de Heston
deltas_bs, gammas_bs, vegas_bs = [], [], []
deltas_he, gammas_he = [], []

dS = S0 * 0.01 # Perturbación del 1% para diferencias finitas

print("Calculando Griegas para ambos modelos (esto puede tomar unos segundos)...")
for t in tiempos:
    # Griegas Black-Scholes analíticas
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma0**2) * t) / (sigma0 * np.sqrt(t))
    deltas_bs.append(norm.cdf(d1))
    gammas_bs.append(norm.pdf(d1) / (S0 * sigma0 * np.sqrt(t)))
    vegas_bs.append(S0 * norm.pdf(d1) * np.sqrt(t) / 100) # Vega por 1%
    
    # Griegas Heston mediante Diferencias Finitas Centrales
    P_up = heston_price(S0 + dS, K, t, r, kappa, theta, nu, v0, rho)
    P_mid = heston_price(S0, K, t, r, kappa, theta, nu, v0, rho)
    P_down = heston_price(S0 - dS, K, t, r, kappa, theta, nu, v0, rho)
    
    deltas_he.append((P_up - P_down) / (2 * dS))
    gammas_he.append((P_up - 2 * P_mid + P_down) / (dS**2))

# Gráficas de Griegas
plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.plot(tiempos, deltas_bs, 'b-', label='BS Delta')
plt.plot(tiempos, deltas_he, 'r--', label='Heston Delta')
plt.title('Evolución de Delta')
plt.gca().invert_xaxis()
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(tiempos, gammas_bs, 'b-', label='BS Gamma')
plt.plot(tiempos, gammas_he, 'r--', label='Heston Gamma')
plt.title('Evolución de Gamma')
plt.gca().invert_xaxis()
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(tiempos, vegas_bs, 'g-', label='BS Vega (por 1%)')
plt.title('Evolución de Vega (BS)')
plt.gca().invert_xaxis()
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================================
# --- PARTE B: Estrategia de Cobertura ---
# ==========================================
print("=== PARTE B: Optimización de Hedging ===")
np.random.seed(42) # Para reproducibilidad

def simulate_delta_hedging(N_steps, cost_rate):
    dt = T / N_steps
    S_path = [S0]
    # Simulamos un camino de precios (GBM)
    for _ in range(N_steps):
        Z = np.random.normal(0, 1)
        S_next = S_path[-1] * np.exp((r - 0.5 * sigma0**2) * dt + sigma0 * np.sqrt(dt) * Z)
        S_path.append(S_next)
    
    total_cost = 0
    prev_delta = 0
    
    # Calculamos el costo de rebalancear en cada paso
    for i in range(N_steps):
        t_remain = T - i * dt
        if t_remain <= 0: break
        d1 = (np.log(S_path[i] / K) + (r + 0.5 * sigma0**2) * t_remain) / (sigma0 * np.sqrt(t_remain))
        current_delta = norm.cdf(d1)
        
        # Costo = (Cambio en Delta) * Precio Acción * Tarifa
        shares_traded = abs(current_delta - prev_delta)
        total_cost += shares_traded * S_path[i] * cost_rate
        prev_delta = current_delta
        
    return total_cost

costo_diario = simulate_delta_hedging(90, c_base) # 90 días
costo_semanal = simulate_delta_hedging(12, c_base) # ~12 semanas

print(f"Costo Total Hedging Diario (0.15%): ${costo_diario:.2f} por opción")
print(f"Costo Total Hedging Semanal (0.15%): ${costo_semanal:.2f} por opción")
print("Conclusión B: El rebalanceo frecuente (diario) minimiza el tracking error pero dispara los costos de transacción.\n")

# ==========================================
# --- PARTE C: Análisis de Sensibilidad ---
# ==========================================
print("=== PARTE C: Análisis de Sensibilidad (Shocks) ===")

# 1. Shock de Volatilidad (+50%)
vol_crisis = sigma0 * 1.5
precio_crisis_bs = black_scholes_call(S0, K, T, r, vol_crisis)
precio_crisis_he = heston_price(S0, K, T, r, kappa, theta, nu, vol_crisis**2, rho)
print(f"1. CRISIS (+50% vol):")
print(f"   Precio BS Crisis: ${precio_crisis_bs:.2f} (Impacto: +${precio_crisis_bs - bs_p:.2f})")
print(f"   Precio Heston Crisis: ${precio_crisis_he:.2f} (Impacto: +${precio_crisis_he - he_p:.2f})")

# 2. Incertidumbre Paramétrica (+/- 20% en Kappa)
kappa_up = kappa * 1.2
kappa_down = kappa * 0.8
precio_kup = heston_price(S0, K, T, r, kappa_up, theta, nu, v0, rho)
precio_kdown = heston_price(S0, K, T, r, kappa_down, theta, nu, v0, rho)
print(f"2. INCERTIDUMBRE KAPPA:")
print(f"   Precio con Kappa +20% (Reversión rápida): ${precio_kup:.2f}")
print(f"   Precio con Kappa -20% (Reversión lenta): ${precio_kdown:.2f}")

# 3. Shock de Costos de Transacción (0.50%)
c_alto = 0.005
costo_diario_alto = simulate_delta_hedging(90, c_alto)
costo_semanal_alto = simulate_delta_hedging(12, c_alto)
print(f"3. SHOCK DE COSTOS (al 0.50%):")
print(f"   Costo Hedging Diario: ${costo_diario_alto:.2f}")
print(f"   Costo Hedging Semanal: ${costo_semanal_alto:.2f}")
print("   Conclusión C3: Ante un aumento de costos, la frecuencia óptima de rebalanceo debe reducirse (moverse a semanal).")