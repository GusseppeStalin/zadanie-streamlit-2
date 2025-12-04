# streamlit_wine_analysis.py
# Rozbudowana aplikacja Streamlit: zaawansowana analityka (3 metody) + predykcja (GradientBoostingRegressor)

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import io
import seaborn as sns

st.set_page_config(page_title="Wine Advanced Analytics", layout="wide")

# -------------------------
# Helpers: find & load files
# -------------------------

def find_file(filename):
    for root, dirs, files in os.walk("."):
        if filename in files:
            return os.path.join(root, filename)
    return None

quality_path = find_file("winequality-red.csv")
food_path = find_file("wine_food_pairings.csv")

if quality_path is None or food_path is None:
    st.error("""Nie znaleziono wymaganych plików CSV w katalogu aplikacji. Upewnij się, że:
- winequality-red.csv
- wine_food_pairings.csv
są obok pliku aplikacji.""")
    if st.button("Pokaż aktualny katalog"):
        st.write(os.listdir("."))
    st.stop()

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

wine = load_csv(quality_path)
pairings = load_csv(food_path)

# -------------------------
# Sidebar: global controls
# -------------------------
st.sidebar.header("Ustawienia aplikacji")
mode = st.sidebar.radio("Sekcja:", ["Zaawansowana analiza", "Predykcja (GradientBoosting)", "Food–Wine Pairings"])

show_raw = st.sidebar.checkbox("Pokaż surowe dane (odpowiedni dataset)")

# -------------------------
# Utility plots
# -------------------------

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    return buf

# -------------------------
# Advanced analysis section
# -------------------------
if mode == "Zaawansowana analiza":
    st.title("Zaawansowana analityka: 3 podejścia")
    st.markdown("""
Porównamy 5 modeli:

- **GradientBoostingRegressor**
- **RandomForestRegressor**
- **ElasticNet**
- **SVR**
- **KNeighborsRegressor**
""")

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor

    numeric = wine.select_dtypes(include=np.number).columns.tolist()
    features = [c for c in numeric if c != 'quality']
    X = wine[features]
    y = wine['quality']

    test_size = st.slider("Test size (%)", 5, 40, 20)
    seed = st.number_input("Random state", 0, 9999, 42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=seed)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "GradientBoosting": GradientBoostingRegressor(random_state=seed),
        "RandomForest": RandomForestRegressor(random_state=seed),
        "ElasticNet": ElasticNet(),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor()
    }

    results = {}
    for name, model in models.items():
        if name in ["ElasticNet", "SVR", "KNN"]:
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        results[name] = {"RMSE": rmse, "R2": r2}

    df_res = pd.DataFrame(results).T

    st.subheader("Wyniki modeli")
    st.dataframe(df_res.style.highlight_min(axis=0, color='lightgreen').highlight_max(axis=0, color='lightblue'))

    st.subheader("Wykres RMSE")
    st.bar_chart(df_res['RMSE'])

    st.subheader("Wykres R²")
    st.bar_chart(df_res['R2'])

# -------------------------
# Prediction: GradientBoosting
# -------------------------
elif mode == "Predykcja (GradientBoosting)":
    st.title("Predykcja jakości wina — GradientBoostingRegressor")

    st.markdown("Użyjemy modelu Gradient Boosting — z opcjonalnym RandomizedSearchCV dla szybkiej optymalizacji hyperparametrów.")

    numeric = wine.select_dtypes(include=np.number).columns.tolist()
    if 'quality' not in numeric:
        st.error('Brak kolumny quality w datasetcie')
        st.stop()

    features = st.multiselect("Wybierz cechy (features)", [c for c in numeric if c != 'quality'], default=[c for c in numeric if c != 'quality'])
    if len(features) < 1:
        st.warning("Wybierz co najmniej jedną cechę.")
        st.stop()

    test_size = st.slider("Test size (%)", 5, 40, 20)
    seed = st.number_input("Random state (seed)", value=42, step=1)

    X = wine[features]
    y = wine['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=seed)

    do_scale = st.checkbox("Standaryzuj cechy przed trenowaniem (zalecane)", value=True)
    if do_scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=features)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=features)

    st.subheader("Definicja modelu")
    base_model = GradientBoostingRegressor(random_state=seed)

    tune = st.checkbox("Użyj RandomizedSearchCV (szybka optymalizacja)")
    if tune:
        param_dist = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [2, 3, 4, 6],
            'subsample': [0.6, 0.8, 1.0],
        }
        n_iter = st.slider("Liczba iteracji RandomizedSearchCV:", 10, 60, 20)
        rs = RandomizedSearchCV(base_model, param_distributions=param_dist, n_iter=n_iter, cv=3, random_state=seed, n_jobs=-1)
        with st.spinner("Optymalizuję hyperparametry..."):
            rs.fit(X_train, y_train)
        model = rs.best_estimator_
        st.success(f"Zakończono RandomizedSearch. Najlepszy zestaw: {rs.best_params_}")
    else:
        n_estimators = st.slider("n_estimators (GB)", 50, 500, 200)
        learning_rate = st.select_slider("learning_rate", options=[0.01, 0.05, 0.1, 0.2], value=0.1)
        max_depth = st.slider("max_depth", 2, 8, 3)
        model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=seed)
        model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse:.3f}")
    col2.metric("R²", f"{r2:.3f}")
    col3.metric("Próbek (train/test)", f"{X_train.shape[0]} / {X_test.shape[0]}")

    st.subheader("Predykcje vs prawda (przykładowe wiersze)")
    comp = pd.DataFrame({'true': y_test.values, 'pred': y_pred})
    st.dataframe(comp.head(20))

    st.subheader("Ważność cech (permutation importance)")
    perm = permutation_importance(model, X_test, y_test, n_repeats=25, random_state=seed, n_jobs=-1)
    perm_imp = pd.Series(perm.importances_mean, index=features).sort_values()
    st.bar_chart(perm_imp)

    st.subheader("Partial dependence plots (pierwsze 2 najważniejsze cechy)")
    top_feats = perm_imp.sort_values(ascending=False).index.tolist()[:2]
    if len(top_feats) >= 1:
        fig, ax = plt.subplots(figsize=(6,4))
        try:
            PartialDependenceDisplay.from_estimator(model, X_test, [top_feats[0]], ax=ax)
            st.pyplot(fig)
        except Exception as e:
            st.write("Nie udało się narysować PDP:", e)
    if len(top_feats) >= 2:
        fig2, ax2 = plt.subplots(figsize=(6,4))
        try:
            PartialDependenceDisplay.from_estimator(model, X_test, [top_feats[1]], ax=ax2)
            st.pyplot(fig2)
        except Exception as e:
            st.write("Nie udało się narysować PDP:", e)

    # Interactive single prediction
    st.subheader("Interaktywna predykcja pojedynczego wina")
    cols = st.columns(3)
    input_dict = {}
    for i, f in enumerate(features):
        col = cols[i % 3]
        min_v = float(wine[f].min())
        max_v = float(wine[f].max())
        med = float(wine[f].median())
        input_dict[f] = col.number_input(f, value=med, min_value=min_v, max_value=max_v, format="%.6f")

    if st.button("Oblicz predykcję jakości dla podanych parametrów"):
        x_single = pd.DataFrame([input_dict])[features]
        if do_scale:
            x_single = pd.DataFrame(scaler.transform(x_single), columns=features)
        pred = model.predict(x_single)[0]
        st.success(f"Przewidywana jakość: {pred:.3f}")

    if st.button("Pobierz model jako pickle"):
        import pickle
        b = io.BytesIO()
        pickle.dump(model, b)
        b.seek(0)
        st.download_button("Pobierz model (.pkl)", data=b, file_name="gbr_model.pkl")

    if show_raw:
        st.markdown("---")
        st.subheader("Surowe dane: winequality-red.csv")
        st.dataframe(wine)

# -------------------------
# Food–Wine Pairings
# -------------------------
elif mode == "Food–Wine Pairings":
    st.title("Food–Wine Pairings — eksploracja i rekomendacje")

    st.markdown("Interaktywna sekcja do filtrowania i prostych rekomendacji.")

    st.subheader("Podgląd danych")
    st.dataframe(pairings.head(50))

    st.subheader("Filtrowanie")
    wine_types = ['Wszystkie'] + sorted(pairings['wine_type'].dropna().unique().tolist())
    sel_wine = st.selectbox("Wybierz typ wina:", wine_types)
    food_cats = st.multiselect("Wybierz kategorię jedzenia:", sorted(pairings['food_category'].dropna().unique().tolist()))

    df_f = pairings.copy()
    if sel_wine != 'Wszystkie':
        df_f = df_f[df_f['wine_type'] == sel_wine]
    if food_cats:
        df_f = df_f[df_f['food_category'].isin(food_cats)]

    st.write(f"Wyników: {len(df_f)}")
    st.dataframe(df_f[['wine_type','food_item','food_category','pairing_quality','cuisine']].sort_values('pairing_quality', ascending=False).head(100))

    st.subheader("Top rekomendacje dla wybranego wina")
    sel_for_rec = st.selectbox("Wybierz wino do rekomendacji:", sorted(pairings['wine_type'].dropna().unique().tolist()))
    top_n = st.slider("Ile rekomendacji?", 1, 20, 5)
    recs = pairings[pairings['wine_type'] == sel_for_rec].sort_values('pairing_quality', ascending=False).head(top_n)
    st.table(recs[['food_item','food_category','cuisine','pairing_quality','quality_label']])

    if show_raw:
        st.markdown("---")
        st.subheader("Surowe dane: wine_food_pairings.csv")
        st.dataframe(pairings)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Aplikacja: zaawansowana analityka + GradientBoostingRegressor. Możesz dalej rozbudować interpretację (SHAP), grid search, lub zapisać artefakty.")
