import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="CRR Binomial Option Pricer", layout="wide")

st.title("CRR Binomial Option Pricer — Application interactive")
st.markdown(
    "Cette application calcule et visualise un arbre binomial (Cox-Ross-Rubinstein) pour pricer des options "
    "européennes et américaines, affiche les deltas de couverture et permet de télécharger les arbres en CSV."
)

# ------------------ Sidebar: paramètres ------------------
st.sidebar.header("Paramètres du modèle")
S0 = st.sidebar.number_input("Prix initial du sous-jacent S0", value=100.0, step=1.0, format="%.2f")
K = st.sidebar.number_input("Prix d'exercice K", value=105.0, step=1.0, format="%.2f")
T = st.sidebar.number_input("Maturité T (années)", value=1.0, step=0.25, format="%.4f")
r = st.sidebar.number_input("Taux sans risque annualisé r", value=0.05, step=0.005, format="%.4f")
sigma = st.sidebar.number_input("Volatilité annualisée sigma", value=0.2, step=0.01, format="%.4f")
n = st.sidebar.slider("Nombre d'étapes n", min_value=1, max_value=50, value=5)

st.sidebar.header("Option & affichage")
option_type = st.sidebar.selectbox("Type d'option", options=["Call", "Put"], index=0)
option_style = st.sidebar.selectbox("Nature de l'option", options=["Européenne", "Américaine"], index=1)
show_prices_tree = st.sidebar.checkbox("Afficher l'arbre des prix", value=True)
show_option_tree = st.sidebar.checkbox("Afficher l'arbre des valeurs (CALL/PUT)", value=True)
show_delta = st.sidebar.checkbox("Afficher la matrice des Deltas", value=True)

# ------------------ Calculs CRR ------------------

dt = T / n
u = np.exp(sigma * np.sqrt(dt))
d = 1.0 / u
p_star = (np.exp(r * dt) - d) / (u - d)

st.sidebar.markdown(f"u = {u:.6f}, d = {d:.6f}, p* = {p_star:.6f}")

# Fonctions utilitaires

def build_price_tree(S0, u, d, n):
    """Construit la matrice des prix S[j,i] où i = étape (0..n) et j = nombre de downs (0..i)."""
    S = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1):
            S[j, i] = S0 * (u ** (i - j)) * (d ** j)
    return S


def price_option_crr(S, K, r, dt, p_star, option_type='Call', american=False):
    n = S.shape[1] - 1
    V = np.zeros_like(S)
    if option_type == 'Call':
        V[:, n] = np.maximum(S[:, n] - K, 0)
    else:
        V[:, n] = np.maximum(K - S[:, n], 0)

    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            cont = np.exp(-r * dt) * (p_star * V[j, i + 1] + (1 - p_star) * V[j + 1, i + 1])
            if american:
                exercise = S[j, i] - K if option_type == 'Call' else K - S[j, i]
                V[j, i] = max(exercise, cont)
            else:
                V[j, i] = cont
    return V


def compute_deltas(V, S):
    # Delta matrix has shape (n, n)
    n = S.shape[1] - 1
    Delta = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            numer = V[j, i + 1] - V[j + 1, i + 1]
            denom = S[j, i + 1] - S[j + 1, i + 1]
            Delta[j, i] = numer / denom
    return Delta


def plot_price_tree(S, u, d, n):
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n):
        for j in range(i + 1):
            x = [i, i + 1]
            y_up = [S[j, i], S[j, i] * u]
            y_down = [S[j, i], S[j, i] * d]
            ax.plot(x, y_up, c='green')
            ax.plot(x, y_down, c='red')

    for i in range(n + 1):
        for j in range(i + 1):
            ax.text(i, S[j, i], f"{S[j, i]:.2f}", ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    ax.set_title("Arbre Binomial — Prix du sous-jacent")
    ax.set_xlabel("Étapes")
    ax.set_ylabel("Prix")
    ax.grid(True)
    return fig


def plot_value_tree(V, title="Arbre des valeurs"):
    n = V.shape[1] - 1
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n):
        for j in range(i + 1):
            x = [i, i + 1]
            y_up = [V[j, i], V[j, i + 1]]
            y_down = [V[j, i], V[j + 1, i + 1]]
            ax.plot(x, y_up)
            ax.plot(x, y_down)

    for i in range(n + 1):
        for j in range(i + 1):
            ax.text(i, V[j, i], f"{V[j, i]:.2f}", ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    ax.set_title(title)
    ax.set_xlabel("Étapes")
    ax.set_ylabel("Valeur")
    ax.grid(True)
    return fig

# ------------------ Run calculations ------------------

S = build_price_tree(S0, u, d, n)
american = True if option_style == 'Américaine' else False
V = price_option_crr(S, K, r, dt, p_star, option_type=option_type, american=american)
Delta = compute_deltas(V, S)

# Prix initial
price_0 = V[0, 0]

# Portefeuille répliquant initial (approx simple)
# On peut répliquer par: acheter Delta[0,0] actions et placer la différence en cash
initial_delta = Delta[0, 0] if Delta.size else np.nan
initial_cash = price_0 - initial_delta * S0

# ------------------ Affichages ------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Résultats")
    st.write(f"**Type d'option :** {option_type} — **{option_style}**")
    st.write(f"**Prix initial (t=0)** : {price_0:.6f}")
    st.write(f"**Delta initial** : {initial_delta:.6f}")
    st.write(f"**Portefeuille répliquant (t=0)** : {initial_delta:.6f} action(s), cash = {initial_cash:.6f}")

    if st.button("Télécharger l'arbre des prix (CSV)"):
        # Préparer DataFrame pour export
        df = pd.DataFrame(S)
        csv = df.to_csv(index=False)
        st.download_button("Télécharger CSV prix", data=csv, file_name="prix_tree.csv", mime="text/csv")

with col2:
    st.subheader("Paramètres")
    st.write({
        "S0": S0,
        "K": K,
        "T": T,
        "r": r,
        "sigma": sigma,
        "n": n,
        "u": float(u),
        "d": float(d),
        "p*": float(p_star),
        "American?": american,
    })

# Afficher matrices sous forme de DataFrame
if st.checkbox("Afficher les matrices sous forme de tables (S, V, Delta)"):
    st.subheader("Matrice des prix S (lignes = j (downs), colonnes = i (steps))")
    st.dataframe(pd.DataFrame(S))
    st.subheader("Matrice des valeurs V")
    st.dataframe(pd.DataFrame(V))
    if show_delta:
        st.subheader("Matrice des Deltas")
        st.dataframe(pd.DataFrame(Delta))

# Graphiques
if show_prices_tree:
    st.subheader("Arbre des prix")
    figS = plot_price_tree(S, u, d, n)
    st.pyplot(figS)

if show_option_tree:
    st.subheader(f"Arbre des valeurs — {option_type}")
    figV = plot_value_tree(V, title=f"Arbre des valeurs — {option_type}")
    st.pyplot(figV)

if show_delta:
    st.subheader("Heatmap des deltas")
    # Afficher heatmap simple avec matplotlib
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(Delta, aspect='auto')
    ax.set_title("Delta matrix (rows = j, cols = i)")
    ax.set_xlabel("Étapes i")
    ax.set_ylabel("j (downs)")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

st.markdown("---")
st.caption("Pour exécuter localement : `pip install streamlit numpy pandas matplotlib` puis `streamlit run streamlit_crr_app.py`")
