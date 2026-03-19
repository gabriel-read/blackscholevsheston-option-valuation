# 📈 Dynamic Option Valuation: Black-Scholes vs. Heston

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive-FF4B4B.svg)
![Quant](https://img.shields.io/badge/Quantitative-Finance-green.svg)

> **🚀 Live Web App:** [Test the Interactive Dashboard Here](https://blackscholevsheston-option-valuation.streamlit.app/)

## 1. Problem Statement
This project focuses on the valuation and risk management of a portfolio consisting of **European Call Options** on a highly volatile technology stock. 

Traditional models like Black-Scholes assume constant volatility, which fails to capture real-world market dynamics such as **volatility clustering** and **mean reversion**. This repository implements an interactive comparison between the classic model and the **Heston Stochastic Volatility Model** to address significant "Vega" risks—losses incurred due to unexpected changes in volatility.

## 2. Theoretical Framework

### Stochastic Volatility Modeling
The system is governed by coupled stochastic processes. While the stock price follows a Geometric Brownian Motion, the volatility itself follows a **mean-reverting Ornstein-Uhlenbeck process**.

### Key Concepts Implemented:
* **Heston Model:** Incorporates stochastic variance to capture the "volatility smile" using numerical integration in the complex plane.
* **Mean Reversion ($\kappa$):** The speed at which volatility returns to its long-term average.
* **Vega Hedging:** Management of portfolio sensitivity to changes in the underlying asset's volatility.
* **Tracking Error vs. Friction:** Analysis of the hedging effectiveness considering transaction costs ($c = 0.15\%$).

## 3. System Parameters
The simulation and valuation in the interactive app are initialized with the following base market data:

| Parameter | Symbol | Value |
| :--- | :--- | :--- |
| Initial Stock Price | $S_0$ | \$120 |
| Strike Price | $K$ | \$125 |
| Time to Maturity | $T$ | 90 Days (0.25 years) |
| Risk-free Rate | $r$ | 4% annual |
| Initial Volatility | $\sigma_0$ | 35% annual |
| Mean Reversion Speed | $\kappa$ | 2.5 |
| Long-term Volatility | $\overline{\sigma}$ | 30% annual |
| Volatility of Volatility | $\nu$ | 0.15 |

## 4. Interactive Analysis and Results

### Comparison of Models & Greeks (Part A)
Quantitative analysis of the valuation gap (Model Risk) between Black-Scholes and Heston. It includes the evolution of the "Greeks" ($\Delta$, $\Gamma$, $V$) over time, visualized dynamically using **Plotly** to demonstrate how stochastic uncertainty flattens the acute risk peaks predicted by classic models.

### Hedging Optimization (Part B)
Evaluation of the optimal **Delta-Hedging** frequency. A Monte Carlo simulation engine projects price paths to minimize tracking error while avoiding capital hemorrhage from transaction costs (comparing Daily vs. Weekly rebalancing).

### Sensitivity Analysis (Part C)
Stress testing the portfolio against:
* **Volatility Shocks:** 50% increase in volatility (crisis events).
* **Parametric Uncertainty:** $\pm 20\%$ error in mean reversion estimates.
* **Illiquid Markets:** Impact of transaction cost increases (up to 0.50%), demonstrating the risk of ruin for high-frequency rebalancing.

## 5. 📂 Theoretical Documentation
To audit the mathematical foundation and the business case behind this codebase, refer to the documents provided in the `/docs/` folder:
* [Mathematical Formulas and PDE Developments](./docs/formulas%20usadas.pdf)
* [Case Study Presentation & Theoretical Framework](./docs/Stochastic_Volatility_Management.pdf)

## 6. Requirements and Local Execution
To run this interactive dashboard locally, ensure you have Python 3.x installed along with the following libraries:
* `streamlit`
* `numpy`
* `scipy`
* `plotly`
* `matplotlib`

**Steps to run:**
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/blackscholevsheston-option-valuation.git](https://github.com/YOUR_USERNAME/blackscholevsheston-option-valuation.git)
