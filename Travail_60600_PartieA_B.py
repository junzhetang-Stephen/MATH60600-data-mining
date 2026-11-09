# Travail en équipe 60600 (A2026) : parties A et B - version script du notebook
# Exécution : python "Travail_60600_PartieA_B.py"  (ou cellule par cellule avec # %%)

try:
    display
except NameError:   # hors Jupyter, display() devient print()
    display = print

# %% [markdown]
# # Travail en équipe 60600 (A2026) : parties A et B
#
# **Partie A** : prédire le premier jeu installé le mois suivant (Y = 1, 2, 3 ou 4), critère = accuracy.
# Point de référence de l'énoncé : forêt `ranger` par défaut, erreur OOB de 28,24 % (accuracy d'environ 71,8 %).
# **Partie B** : optimisation des hyperparamètres, appliquée aux données de jeux en ligne du cours (Y continu).
#
# Le notebook s'exécute de haut en bas (*Restart & Run All*). Durée indicative : 10 à 20 min,
# surtout pour les étapes Optuna (on peut baisser `N_ESSAIS_A` et `N_ESSAIS_B` pour un essai rapide).

# %%
# ============================================================================
# PACKAGES REQUIS (Python >= 3.9)
#   pandas, numpy, matplotlib, scikit-learn (>= 1.4), optuna (>= 3.0)
#   Installation au besoin : pip install pandas numpy matplotlib scikit-learn optuna
# ============================================================================
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              HistGradientBoostingClassifier, HistGradientBoostingRegressor)
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)   # n'affiche pas chaque essai
print("scikit-learn", sklearn.__version__, "| optuna", optuna.__version__)

DOSSIER = "./"   # dossier contenant train.csv, test.csv et clv_mobile_data_*.txt
SEED = 60600     # graine unique : tous les résultats sont reproductibles
CATEG = ["country", "difflevel"]   # variables traitées comme catégorielles, comme dans le cours

# %% [markdown]
# ---
# # Partie A : classification à 4 classes
#
# ## A1. Données

# %%
# Charger les données
train = pd.read_csv(DOSSIER + "train.csv")
test = pd.read_csv(DOSSIER + "test.csv")

# Afficher les dimensions
print("Train :", train.shape)
print("Test :", test.shape)

# Afficher les premières lignes
display(train.head())
display(test.head())

# %%
# Distribution de la variable cible Y
display(pd.DataFrame({"n": train["y"].value_counts().sort_index(),
                      "proportion": train["y"].value_counts(normalize=True).sort_index().round(3)}))

# %%
# Statistiques descriptives et valeurs manquantes
display(train.describe().T.round(3))
print("Valeurs manquantes - train :", int(train.isna().sum().sum()), "| test :", int(test.isna().sum().sum()))

# %% [markdown]
# Les 4 classes ont des fréquences presque égales (environ 25 % chacune) : l'accuracy est donc un critère
# adéquat et le modèle naïf ne fait qu'environ 25 %. Par contre, en **OVA** chaque problème binaire oppose
# une classe à trois autres (environ 25 % contre 75 %), le déséquilibre que Galar et al. (2011) identifient
# comme la principale faiblesse d'OVA. En **OVO**, chaque paire reste équilibrée (environ 50/50).
#
# Comme dans le cours, `country` (6 régions) et `difflevel` (3 niveaux) sont traitées comme catégorielles par le boosting.

# %%
# Préparer les variables explicatives
def categ(df, refs):
    """Variables explicatives avec country et difflevel en type 'category'.
    Les modalités sont fixées à partir de refs (train + test) pour être identiques partout."""
    X = df.drop(columns=[c for c in ("y", "id") if c in df]).copy()
    for c in CATEG:
        X[c] = pd.Categorical(X[c], categories=sorted(set().union(*(r[c] for r in refs))))
    return X

def num(X):
    """Version entièrement numérique (codes entiers) pour la forêt aléatoire de scikit-learn."""
    return X.astype({c: int for c in CATEG})

X, y = categ(train, [train, test]), train["y"]
X_test = categ(test, [train, test])
CLASSES = sorted(y.unique())
print("Variables explicatives :", list(X.columns))
print("Classes :", CLASSES)

# %% [markdown]
# ## A2. Protocole d'évaluation
# On combine les deux approches vues en classe (division de l'échantillon + validation croisée) :
#
# 1. **1 000 observations mises de côté** (« test interne ») : elles servent **une seule fois**, à la fin,
#    pour confirmer la performance des modèles retenus ;
# 2. sur les **4 000 autres**, une **validation croisée stratifiée à 5 plis** sert à comparer les méthodes et à
#    optimiser les hyperparamètres. Toutes les méthodes utilisent **les mêmes plis** (comparaison appariée).
#
# On rapporte la moyenne et l'erreur-type (ET) de l'accuracy entre les plis.
# Avec 1 000 observations, l'erreur-type d'une accuracy d'environ 0,8 est d'environ √(0,8×0,2/1000) ≈ 1,3 point :
# des écarts de quelques dixièmes de point entre deux méthodes ne sont pas significatifs.

# %%
# Diviser : 4 000 obs. de développement / 1 000 obs. de test interne (stratifié selon Y)
X_dev, X_hold, y_dev, y_hold = train_test_split(X, y, test_size=1000, stratify=y, random_state=SEED)
print("Développement :", X_dev.shape, "| Test interne :", X_hold.shape)

# Créer les 5 plis de validation croisée (les mêmes pour toutes les méthodes)
PLIS = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_dev, y_dev))
print("Taille des plis de validation :", [len(va) for tr, va in PLIS])

# %%
# Fonctions d'évaluation par validation croisée
def val_croisee(methode, params=None):
    """Validation croisée à 5 plis sur les 4 000 obs. de développement.
    methode(X_tr, y_tr, X_va, params) retourne un dict {nom: prédictions}.
    Retourne un dict {nom: liste des 5 accuracy}."""
    acc = {}
    for tr, va in PLIS:
        preds = methode(X_dev.iloc[tr], y_dev.iloc[tr], X_dev.iloc[va], params)
        for nom, p in preds.items():
            acc.setdefault(nom, []).append(accuracy_score(y_dev.iloc[va], p))
    return acc

def resume(acc):
    """Tableau : accuracy moyenne en VC et son erreur-type."""
    return pd.DataFrame({"acc_VC": {n: np.mean(v) for n, v in acc.items()},
                         "ET": {n: np.std(v, ddof=1) / np.sqrt(len(v)) for n, v in acc.items()}})

resultats = {}   # accumule les accuracy par pli de toutes les méthodes

# %% [markdown]
# ## A3. Points de référence : modèle naïf et forêt aléatoire directe
# Pour la forêt aléatoire, `mtry` (`max_features`) est généralement **l'hyperparamètre le plus important**,
# alors que le nombre d'arbres n'a pas à être optimisé : il suffit d'en avoir assez (notes de cours). On le choisit
# avec l'**erreur OOB**, sans avoir à diviser l'échantillon. 500 arbres et des feuilles de taille 1 correspondent
# au `ranger` par défaut de l'énoncé.

# %%
# Modèle naïf : prédit toujours la classe la plus fréquente de l'échantillon d'entrainement
def naif(X_tr, y_tr, X_va, params):
    return {"Naïf (classe majoritaire)": np.full(len(X_va), y_tr.mode()[0])}

resultats.update(val_croisee(naif))
display(resume(resultats).round(4))

# %%
# Forêt aléatoire : choisir mtry avec l'accuracy OOB (sur les 4 000 obs. de développement)
oob = {}
for m in [2, 3, 4, 6, 8, 10, 12, 16]:
    rf_m = RandomForestClassifier(n_estimators=500, max_features=m, oob_score=True,
                                  n_jobs=-1, random_state=SEED).fit(num(X_dev), y_dev)
    oob[m] = rf_m.oob_score_

display(pd.DataFrame({"accuracy OOB": oob}).rename_axis("mtry").round(4))
MTRY = max(oob, key=oob.get)
print("mtry retenu :", MTRY)

# %%
# Forêt aléatoire directe : validation croisée avec mtry=4 (défaut) et mtry choisi par OOB
def foret(X_tr, y_tr, X_va, params):
    sortie = {}
    for m, nom in [(4, "RF direct (mtry=4, défaut)"), (MTRY, f"RF direct (mtry={MTRY}, OOB)")]:
        rf_m = RandomForestClassifier(n_estimators=500, max_features=m, n_jobs=-1, random_state=SEED)
        sortie[nom] = rf_m.fit(num(X_tr), y_tr).predict(num(X_va))
    return sortie

resultats.update(val_croisee(foret))
display(resume(resultats).round(4))

# %% [markdown]
# ## A4. Boosting : approche directe ou décomposition binaire (Galar et al., 2011)
# On compare trois stratégies avec **le même classifieur de base** (boosting d'arbres) et **les mêmes
# hyperparamètres de départ** `P0`, comme dans l'étude de Galar et al. Le classifieur de base est
# `HistGradientBoosting` de scikit-learn, qui suit le même principe que LightGBM utilisé en classe
# (histogrammes, arbres construits feuille par feuille) et traite directement les variables catégorielles.
#
# * **Direct** : un seul modèle de boosting multi-classes (perte multinomiale) ;
# * **OVA** (*one-vs-all*) : 4 modèles « classe k contre les autres ». On prédit la classe dont le modèle
#   donne la plus grande confiance (agrégation **MAX**) ;
# * **OVO** (*one-vs-one*) : 6 modèles, un par paire de classes (a, b), chacun entrainé seulement sur les
#   observations des classes a et b. Deux agrégations de l'article :
#   **VOTE** (chaque modèle vote pour une classe) et **WV** (vote pondéré : on additionne les confiances
#   r_ab et 1 − r_ab), que l'article classe parmi les agrégations les plus robustes.
#
# Dans le boosting, le nombre d'itérations doit être optimisé (risque de sur-ajustement), le taux
# d'apprentissage doit simplement être petit, et la taille des arbres a souvent un impact important (notes de cours).
# Ici, le nombre d'itérations est fixé par **arrêt précoce sur 10 % de l'échantillon d'entrainement** : c'est
# l'idée de l'exemple LightGBM du cours (choisir les itérations sur une partie validation), mais faite à
# l'intérieur de chaque entrainement. Le pli de validation n'est jamais touché, donc **pas de fuite d'information**.

# %%
# Classifieur de base : boosting d'arbres avec arrêt précoce
def booster(params, regression=False):
    """Boosting d'arbres (HistGradientBoosting, même famille d'algorithme que LightGBM).
    - country et difflevel sont traitées directement comme catégorielles (from_dtype) ;
    - le nombre d'itérations est choisi par arrêt précoce sur 10 % de l'échantillon
      d'ENTRAINEMENT : le pli de validation n'est jamais utilisé pour l'ajuster."""
    classe = HistGradientBoostingRegressor if regression else HistGradientBoostingClassifier
    return classe(**params, max_iter=2000, early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=20, categorical_features="from_dtype", random_state=SEED)

# Hyperparamètres de départ, identiques pour les 3 stratégies
P0 = dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20,
          l2_regularization=0.0, max_features=1.0)
print("P0 :", P0)

# %%
# Stratégie 1 : boosting direct (un seul modèle à 4 classes)
def direct(X_tr, y_tr, X_va, params):
    return {"Boosting direct": booster(params).fit(X_tr, y_tr).predict(X_va)}

resultats.update(val_croisee(direct, P0))
display(resume(resultats).round(4))

# %%
# Stratégie 2 : OVA, 4 modèles « classe k contre les 3 autres », agrégation MAX
def ova(X_tr, y_tr, X_va, params):
    # Colonne k = confiance que l'observation appartient à la classe k
    conf = np.column_stack([booster(params).fit(X_tr, y_tr == k).predict_proba(X_va)[:, 1]
                            for k in CLASSES])
    return {"Boosting OVA (MAX)": np.array(CLASSES)[conf.argmax(axis=1)]}

resultats.update(val_croisee(ova, P0))
display(resume(resultats).round(4))

# %%
# Stratégie 3 : OVO, un modèle binaire par paire de classes (6 modèles)
def ajuster_ovo(X_tr, y_tr, params):
    # Le modèle (a, b) est entrainé seulement sur les obs. des classes a et b
    modeles = {}
    for a, b in combinations(CLASSES, 2):
        garder = y_tr.isin([a, b])
        modeles[(a, b)] = booster(params).fit(X_tr[garder], y_tr[garder] == a)
    return modeles

def predire_ovo(modeles, X_va, agregation="WV"):
    """Agrégation des 6 modèles (Galar et al., 2011) : VOTE ou WV (vote pondéré)."""
    score = pd.DataFrame(0.0, index=range(len(X_va)), columns=CLASSES)
    for (a, b), mod in modeles.items():
        r_ab = mod.predict_proba(X_va)[:, 1]          # confiance « classe a plutôt que b »
        if agregation == "VOTE":
            r_ab = (r_ab > 0.5).astype(float)         # vote 0/1
        score[a] += r_ab
        score[b] += 1 - r_ab
    return score.idxmax(axis=1).to_numpy()            # classe avec le plus grand score

def ovo(X_tr, y_tr, X_va, params):
    modeles = ajuster_ovo(X_tr, y_tr, params)
    return {"Boosting OVO (VOTE)": predire_ovo(modeles, X_va, "VOTE"),
            "Boosting OVO (WV)": predire_ovo(modeles, X_va, "WV")}

# %%
# Validation croisée de l'OVO (les deux agrégations utilisent les mêmes 6 modèles)
resultats.update(val_croisee(ovo, P0))
display(resume(resultats).round(4))

# %% [markdown]
# ## A4 bis. Chercher la structure : variables dérivées et modèle paramétrique
# Des équipes atteignent environ 88 % sur le tableau public, ce qui veut dire qu'il reste du signal que le
# boosting sur les 16 variables brutes ne capte pas. Les données sont simulées : si Y dépend de **ratios** ou
# d'**interactions**, les arbres doivent les approximer par une série de coupures, ce qui demande beaucoup
# d'observations. Deux pistes inspirées du cours :
#
# 1. **Variables dérivées** (ratios « par unité ») ajoutées aux 16 variables, par exemple le montant moyen par achat.
#    Dans l'exemple des jeux du cours, l'interaction la plus forte (H-index) est `numpurchases` × `totpurchases`.
# 2. **Régression logistique multinomiale avec les carrés et les interactions d'ordre 2**, l'équivalent pour Y
#    catégoriel du benchmark « Rég lin (carrés et inter d'ordre 2) » du cours, qui fait un très grand saut par rapport
#    à la régression simple. Avec plusieurs centaines de termes, on ajoute une pénalité ridge (L2) dont la force
#    (paramètre C) est choisie par validation croisée. Les variables sont **standardisées avec les moyennes et
#    écarts-types de l'échantillon d'entrainement seulement** (le pipeline est réajusté dans chaque pli), comme le
#    recommande le cours pour ne pas contaminer les données de validation.

# %%
# Créer les variables dérivées (ratios « par unité »)
def derivees(X):
    """Ajoute 7 ratios aux 16 variables (dénominateur ramené à au moins 1)."""
    X = X.copy()
    ratio = lambda a, b: X[a] / X[b].clip(lower=1)
    X["achat_moyen"] = ratio("totpurchases", "numpurchases")     # montant moyen par achat
    X["achats_session"] = ratio("numpurchases", "numsessions")   # achats par session
    X["sessions_jour"] = ratio("numsessions", "numdays")         # sessions par jour joué
    X["temps_session"] = ratio("totplaytime", "numsessions")     # minutes par session
    X["temps_jour"] = ratio("totplaytime", "numdays")            # minutes par jour joué
    X["vies_element"] = ratio("numlives", "numelements")         # vies utilisées par niveau réussi
    X["score_element"] = ratio("totscore", "numelements")        # score par niveau réussi
    return X

def avec_derivees(methode):
    """Même méthode, mais appliquée aux variables originales + dérivées."""
    def f(X_tr, y_tr, X_va, params):
        preds = methode(derivees(X_tr), y_tr, derivees(X_va), params)
        return {nom + " + dérivées": p for nom, p in preds.items()}
    return f

# Afficher les nouvelles variables
display(derivees(X).iloc[:, -7:].head().round(3))

# %%
# Boosting (3 stratégies) avec les variables dérivées, mêmes hyperparamètres P0
for methode in (direct, ova, ovo):
    resultats.update(val_croisee(avec_derivees(methode), P0))
display(resume(resultats).round(4))

# %%
# Régression logistique multinomiale avec carrés et interactions d'ordre 2
def nom_logit(params):
    return f"Logistique carrés+inter. (C={params['C']})" + (" + dérivées" if params["derivees"] else "")

def logistique(X_tr, y_tr, X_va, params):
    """Indicatrices pour country et difflevel, standardisation, puis tous les carrés et
    interactions d'ordre 2, standardisés à nouveau, et pénalité ridge de force 1/C."""
    transfo = derivees if params["derivees"] else (lambda Z: Z)
    modele = make_pipeline(
        ColumnTransformer([("cat", OneHotEncoder(drop="first"), CATEG)], remainder=StandardScaler()),
        PolynomialFeatures(degree=2, include_bias=False), StandardScaler(),
        LogisticRegression(C=params["C"], max_iter=5000))
    return {nom_logit(params): modele.fit(transfo(X_tr), y_tr).predict(transfo(X_va))}

# %%
# Recherche en grille sur C, avec et sans variables dérivées
P_LOGIT = [{"C": C, "derivees": d} for d in (False, True) for C in (0.001, 0.01, 0.1, 1.0)]
for p in P_LOGIT:
    resultats.update(val_croisee(logistique, p))
display(resume({nom_logit(p): resultats[nom_logit(p)] for p in P_LOGIT}).round(4))

# %%
# Résumé de toutes les méthodes comparées jusqu'ici (triées par accuracy en VC)
display(resume(resultats).sort_values("acc_VC", ascending=False).round(4))

# %% [markdown]
# **Importance des variables par permutation** (méthode du cours) pour le boosting direct avec variables dérivées :
# le modèle est entrainé sur les 4 plis d'entrainement du premier découpage, puis on mesure la perte d'accuracy
# sur le pli de validation quand on permute aléatoirement chaque variable. Les variables du haut indiquent où
# chercher d'autres transformations utiles.

# %%
# Importance des variables par permutation
tr, va = PLIS[0]
mod_vimp = booster(P0).fit(derivees(X_dev.iloc[tr]), y_dev.iloc[tr])
vimp = permutation_importance(mod_vimp, derivees(X_dev.iloc[va]), y_dev.iloc[va], scoring="accuracy",
                              n_repeats=5, random_state=SEED)
vimp = pd.Series(vimp.importances_mean, index=derivees(X_dev).columns).sort_values()
vimp.plot.barh(figsize=(6, 6), title="Importance par permutation (baisse d'accuracy)")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## A5. Optimisation des hyperparamètres (lien avec la partie B)
# On optimise les deux stratégies de boosting finalistes, **direct** et **OVO (WV)**, chacune avec ou sans
# variables dérivées selon ce qui a donné la meilleure VC en A4 bis. Les deux ont la même méthode et le même budget :
# **optimisation bayésienne** (échantillonneur TPE d'Optuna, `N_ESSAIS_A` essais chacun). Le critère est
# l'accuracy moyenne de la validation croisée à 5 plis ci-dessus. Pour OVO, un même jeu d'hyperparamètres est
# utilisé pour les 6 modèles binaires (chacun garde son propre nombre d'itérations par arrêt précoce).
#
# Comme les hyperparamètres sont choisis en maximisant cette accuracy de VC, celle du meilleur essai est
# légèrement **optimiste**. C'est pourquoi on confirme ensuite sur le test interne (A6).

# %%
# Espace de recherche des hyperparamètres du boosting (utilisé aussi en partie B)
def espace(trial):
    return dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),    # rétrécissement
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 4, 64, log=True),        # taille des arbres
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 5, 100, log=True),   # taille min. d'une feuille
        l2_regularization=trial.suggest_float("l2_regularization", 1e-4, 10, log=True),
        max_features=trial.suggest_float("max_features", 0.3, 1.0))                 # analogue de mtry

# Pour chaque stratégie, garder la version (avec ou sans variables dérivées) qui a la meilleure VC
acc_moy = lambda nom: np.mean(resultats[nom])

def version_choisie(nom, methode):
    if acc_moy(nom + " + dérivées") > acc_moy(nom):
        return nom + " + dérivées", avec_derivees(methode)
    return nom, methode

FIN_DIRECT = version_choisie("Boosting direct", direct)
FIN_OVO = version_choisie("Boosting OVO (WV)", ovo)
print("Finalistes :", FIN_DIRECT[0], "|", FIN_OVO[0])

# %%
# Fonction d'optimisation bayésienne (TPE) d'une méthode
N_ESSAIS_A = 25
OPTIMISES = {}   # nom -> (méthode, meilleurs hyperparamètres)

def optimiser(nom, methode):
    def objectif(trial):
        acc = [float(a) for a in val_croisee(methode, espace(trial))[nom]]
        trial.set_user_attr("acc_plis", acc)       # garde l'accuracy de chaque pli
        return float(np.mean(acc))
    etude = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    etude.optimize(objectif, n_trials=N_ESSAIS_A)
    resultats[nom + " optimisé"] = etude.best_trial.user_attrs["acc_plis"]
    OPTIMISES[nom + " optimisé"] = (methode, etude.best_params)
    print(f"{nom} : meilleure acc. VC = {etude.best_value:.4f}")
    print("Meilleurs hyperparamètres :", {k: round(v, 4) for k, v in etude.best_params.items()})
    return etude

# %%
# Optimiser le boosting direct
etude_direct = optimiser(*FIN_DIRECT)

# %%
# Optimiser le boosting OVO (WV)
etude_ovo = optimiser(*FIN_OVO)

# %%
# Évolution de la meilleure accuracy trouvée en fonction du nombre d'essais
for nom, etude in [(FIN_DIRECT[0], etude_direct), (FIN_OVO[0], etude_ovo)]:
    plt.plot(np.maximum.accumulate([t.value for t in etude.trials]), label=nom)
plt.xlabel("Nombre d'essais"); plt.ylabel("Meilleure accuracy en VC (5 plis)")
plt.title("Optimisation bayésienne (TPE)"); plt.legend(); plt.show()

# %% [markdown]
# ## A6. Comparaison finale et confirmation sur le test interne
# Chaque méthode est réentrainée sur les 4 000 observations de développement, puis évaluée **une seule fois**
# sur les 1 000 observations mises de côté (`acc_test_interne`). Le modèle final est celui qui a la meilleure
# accuracy de VC parmi les candidats (les deux boostings optimisés et la meilleure logistique) : le test interne
# ne sert pas à choisir, seulement à confirmer.

# %%
# Réentrainer chaque configuration sur les 4 000 obs. de développement et l'évaluer sur le test interne
configs = ([(naif, None), (foret, None)] + [(m, P0) for m in (direct, ova, ovo)]
           + [(avec_derivees(m), P0) for m in (direct, ova, ovo)] + [(logistique, p) for p in P_LOGIT])
pred_hold = {}
for methode, params in configs:
    pred_hold.update(methode(X_dev, y_dev, X_hold, params))
for nom, (methode, params) in OPTIMISES.items():
    pred_hold[nom] = methode(X_dev, y_dev, X_hold, params)[nom.replace(" optimisé", "")]

# Tableau récapitulatif : accuracy en VC, erreur-type et accuracy sur le test interne
tableau_A = resume(resultats)
tableau_A["acc_test_interne"] = pd.Series({n: accuracy_score(y_hold, p) for n, p in pred_hold.items()})
display(tableau_A.sort_values("acc_VC", ascending=False).round(4))

# %%
# Choisir le modèle final : meilleure accuracy en VC parmi les candidats
candidats = dict(OPTIMISES)
P_LOGIT_MEILLEUR = max(P_LOGIT, key=lambda p: acc_moy(nom_logit(p)))   # C choisi par VC
candidats[nom_logit(P_LOGIT_MEILLEUR)] = (logistique, P_LOGIT_MEILLEUR)
display(tableau_A.loc[list(candidats)].sort_values("acc_VC", ascending=False).round(4))

NOM_FINAL = max(candidats, key=lambda n: tableau_A.loc[n, "acc_VC"])
print("Modèle retenu :", NOM_FINAL)

# %%
# Matrice de confusion du modèle retenu sur le test interne (lignes = vraie classe, colonnes = prédite)
mc = confusion_matrix(y_hold, pred_hold[NOM_FINAL], labels=CLASSES)
display(pd.DataFrame(mc, index=[f"vrai {k}" for k in CLASSES], columns=[f"prédit {k}" for k in CLASSES]))
print("Accuracy par classe :", {k: round(mc[i, i] / mc[i].sum(), 3) for i, k in enumerate(CLASSES)})

# %% [markdown]
# ## A7. Modèle final et fichier de soumission Kaggle
# Le modèle retenu est réentrainé avec **les 5 000 observations** (mêmes hyperparamètres) puis appliqué aux
# 55 000 observations de `test.csv`. Une seule soumission finale : le tableau public (9 % des données)
# ne doit pas servir à choisir le modèle, pour éviter le sur-ajustement.

# %%
# Réentrainer le modèle retenu sur les 5 000 obs. et prédire les 55 000 obs. de test.csv
methode, P_FINAL = candidats[NOM_FINAL]
print("Modèle final :", NOM_FINAL)
print("Hyperparamètres :", {k: round(v, 4) if isinstance(v, float) else v for k, v in P_FINAL.items()})
pred_test = methode(X, y, X_test, P_FINAL)[NOM_FINAL.replace(" optimisé", "")]

# %%
# Créer le fichier de soumission Kaggle (colonnes id et y)
soumission = pd.DataFrame({"id": test["id"], "y": pred_test.astype(int)})
soumission.to_csv(DOSSIER + "soumission.csv", index=False)

print("Dimensions :", soumission.shape)
display(soumission.head())
display(soumission["y"].value_counts(normalize=True).sort_index().round(3).to_frame("proportion prédite"))

# %% [markdown]
# ---
# # Partie B : optimisation des hyperparamètres avec les données du cours (Y continu)
#
# Données de l'exemple de jeux en ligne du cours : Y = montant dépensé de 8 à 190 jours après l'installation,
# 5 000 observations d'entrainement et 5 000 observations test (utilisées seulement à la fin, comme en classe).
# Modèle : boosting d'arbres pour la régression, le modèle le plus performant de l'exemple en classe.
#
# Valeurs de référence obtenues en classe (R) sur le même échantillon test : RF `ranger` MSE = 38,9 ;
# boosting LightGBM (135 itérations) MSE = 25,0 et R² = 0,835.

# %%
# Charger les données du cours (fichiers fournis dans Zone Cours)
mob_tr = pd.read_csv(DOSSIER + "clv_mobile_data_train.txt", sep=r"\s+")
mob_te = pd.read_csv(DOSSIER + "clv_mobile_data_test.txt", sep=r"\s+")

# Afficher les dimensions et les premières lignes
print("Train :", mob_tr.shape)
print("Test :", mob_te.shape)
display(mob_tr.head())
display(mob_tr["y"].describe().to_frame("y (train)").T.round(3))

# %%
# Préparer les variables (country et difflevel catégorielles) et la fonction de performance
Xb, yb = categ(mob_tr, [mob_tr, mob_te]), mob_tr["y"]
Xb_te, yb_te = categ(mob_te, [mob_tr, mob_te]), mob_te["y"]
ybar = yb.mean()

def perfcont(y_vrai, pred, ybartrain, nom):
    """Traduction en Python de la fonction perfcont du cours : MSE, R2 hors échantillon, MAE, MAPE."""
    e = y_vrai - pred
    return pd.DataFrame({"MSE": [np.mean(e ** 2)],
                         "R2": [1 - np.sum(e ** 2) / np.sum((y_vrai - ybartrain) ** 2)],
                         "MAE": [np.mean(np.abs(e))],
                         "MAPE": [100 * np.mean(np.abs(e) / np.abs(y_vrai))]}, index=[nom])

# %%
# Points de référence : moyenne de Y et forêt aléatoire (mtry=4, comme ranger par défaut)
res_B = [perfcont(yb_te, ybar, ybar, "Moyenne de Y")]
rf_b = RandomForestRegressor(n_estimators=500, max_features=4, min_samples_leaf=5,
                             n_jobs=-1, random_state=SEED).fit(num(Xb), yb)
res_B.append(perfcont(yb_te, rf_b.predict(num(Xb_te)), ybar, "RF (mtry=4, comme ranger par défaut)"))
display(pd.concat(res_B).round(3))

# %%
# Boosting « comme en classe » : paramètres par défaut (taux 0,1, 31 feuilles, 20 obs. par feuille,
# comme LightGBM) ; nombre d'itérations choisi sur une division 4 000 (entrainement) / 1 000 (validation)
Xb_tt, Xb_va, yb_tt, yb_va = train_test_split(Xb, yb, test_size=1000, random_state=SEED)
gb_400 = HistGradientBoostingRegressor(max_iter=400, early_stopping=False, categorical_features="from_dtype",
                                       random_state=SEED).fit(Xb_tt, yb_tt)
mse_iter = [np.mean((yb_va - p) ** 2) for p in gb_400.staged_predict(Xb_va)]
M = int(np.argmin(mse_iter)) + 1
print("Nombre d'itérations retenu :", M)

plt.plot(range(1, 401), mse_iter); plt.axvline(M, ls="--")
plt.xlabel("Nombre d'itérations"); plt.ylabel("MSE de validation"); plt.title("Choix du nombre d'itérations")
plt.show()

# %%
# Réajuster le boosting par défaut sur les 5 000 obs. avec M itérations, puis évaluer sur le test
gb_M = HistGradientBoostingRegressor(max_iter=M, early_stopping=False, categorical_features="from_dtype",
                                     random_state=SEED).fit(Xb, yb)
res_B.append(perfcont(yb_te, gb_M.predict(Xb_te), ybar, f"Boosting par défaut ({M} itér., comme en classe)"))
display(pd.concat(res_B).round(3))

# %% [markdown]
# ## Application : recherche aléatoire vs optimisation bayésienne
# **Méthode appliquée : optimisation bayésienne (TPE, Optuna)**, comparée à une **recherche aléatoire** avec
# exactement le même espace (fonction `espace` de la partie A), le même budget (`N_ESSAIS_B` essais) et les mêmes plis.
# Le critère à minimiser est le MSE en validation croisée à 5 plis sur l'échantillon d'entrainement. Le TPE commence
# par 10 essais aléatoires, puis propose des combinaisons dans les régions où les essais précédents ont été les meilleurs.

# %%
# Fonction objectif : MSE moyen en validation croisée à 5 plis (à minimiser)
N_ESSAIS_B = 40
PLIS_B = KFold(n_splits=5, shuffle=True, random_state=SEED)

def objectif_B(trial):
    modele = booster(espace(trial), regression=True)
    return -cross_val_score(modele, Xb, yb, cv=PLIS_B, scoring="neg_mean_squared_error").mean()

# %%
# Recherche aléatoire
etude_alea = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(seed=SEED))
etude_alea.optimize(objectif_B, n_trials=N_ESSAIS_B)
print("Recherche aléatoire : meilleur MSE VC =", round(etude_alea.best_value, 3))

# %%
# Optimisation bayésienne (TPE)
etude_tpe = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
etude_tpe.optimize(objectif_B, n_trials=N_ESSAIS_B)
print("Optimisation bayésienne (TPE) : meilleur MSE VC =", round(etude_tpe.best_value, 3))

# %%
# Comparer les deux méthodes : meilleur MSE trouvé en fonction du nombre d'essais
etudes_B = {"Recherche aléatoire": etude_alea, "Optimisation bayésienne (TPE)": etude_tpe}
for nom, etude in etudes_B.items():
    plt.plot(np.minimum.accumulate([t.value for t in etude.trials]), label=nom)
plt.xlabel("Nombre d'essais"); plt.ylabel("Meilleur MSE en VC (5 plis)")
plt.title("Recherche aléatoire vs optimisation bayésienne"); plt.legend(); plt.show()

# Meilleurs hyperparamètres trouvés par chaque méthode
display(pd.DataFrame({nom: {**e.best_params, "MSE_VC": e.best_value} for nom, e in etudes_B.items()}).round(4))

# %% [markdown]
# **Performance sur les 5 000 observations test.** Chaque modèle optimisé est réentrainé sur les 5 000
# observations d'entrainement avec ses meilleurs hyperparamètres, puis évalué une seule fois sur le test.

# %%
# Évaluer les modèles optimisés sur l'échantillon test et comparer aux points de référence
for nom, etude in etudes_B.items():
    modele = booster(etude.best_params, regression=True).fit(Xb, yb)
    res_B.append(perfcont(yb_te, modele.predict(Xb_te), ybar, f"Boosting + {nom}"))
display(pd.concat(res_B).round(3))
