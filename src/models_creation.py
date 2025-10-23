

"""
==============================================================================
FONCTIONS DE MODÉLISATION - DÉTECTION DE FRAUDE
==============================================================================
1. train_supervised_models() : Modèles supervisés avec SMOTE + GridSearchCV
2. train_anomaly_models() : Modèles de détection d'anomalies
3. save_models() : Sauvegarde des modèles entraînés
==============================================================================
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_auc_score, 
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
import warnings
warnings.filterwarnings('ignore')


def train_supervised_models(X_train, y_train, 
                            test_size=0.2,
                            random_state=42,
                            n_jobs=-1):
    """
    Entraîne des modèles supervisés avec SMOTE et GridSearchCV.
    
    Modèles : Logistic Regression, Random Forest, XGBoost
    
    Pipeline :
    1. Split train → train/validation
    2. Application de SMOTE sur train uniquement
    3. GridSearchCV pour chaque modèle
    4. Évaluation sur validation (données réelles, sans SMOTE)
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Features d'entraînement (déjà prétraitées)
    y_train : pd.Series
        Variable cible
    test_size : float, default=0.2
        Proportion pour validation
    random_state : int, default=42
        Seed pour reproductibilité
    n_jobs : int, default=-1
        Nombre de CPU à utiliser
    
    Returns:
    --------
    results : dict
        Dictionnaire contenant les modèles entraînés et leurs performances
        {
            'models': { 'RandomForest': model, 'XGBoost': model},
            'best_params': {...},
            'scores': {...},
            'predictions': {...}
        }
    """
    
    print("=" * 80)
    print("ENTRAÎNEMENT DES MODÈLES SUPERVISÉS")
    print("=" * 80)
    
    
    # 1. SPLIT TRAIN / VALIDATION
  
    print(f"\n Split des données (train {1-test_size:.0%} / validation {test_size:.0%})...")
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train
    )
    
    print(f" Train : {X_tr.shape[0]} transactions ({y_tr.sum()} fraudes)")
    print(f" Validation : {X_val.shape[0]} transactions ({y_val.sum()} fraudes)")
    
    
    # 2. APPLICATION DE SMOTE SUR TRAIN UNIQUEMENT
   
    print("\n Application de SMOTE sur le train set...")
    print(f"  Avant SMOTE : {y_tr.value_counts().to_dict()}")
    
    smote = SMOTE(random_state=random_state, n_jobs=n_jobs)
    X_tr_balanced, y_tr_balanced = smote.fit_resample(X_tr, y_tr)
    
    print(f"  Après SMOTE : {pd.Series(y_tr_balanced).value_counts().to_dict()}")
    print(f" Données équilibrées : {X_tr_balanced.shape[0]} transactions")
    
    
    # 3. DÉFINITION DES MODÈLES ET GRIDS
    
    print("\n Configuration des modèles et GridSearchCV...")
    
    models_config = {
   
        'RandomForest': {
            'model': RandomForestClassifier(random_state=random_state, n_jobs=n_jobs),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=random_state, n_jobs=n_jobs, eval_metric='logloss'),
            'params': {
                'n_estimators': [100,150 ,200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        }
    }
    
    results = {
        'models': {},
        'best_params': {},
        'scores': {},
        'predictions': {}
    }
    
  
    # 4. ENTRAÎNEMENT AVEC GRIDSEARCHCV
    
    print("\n Entraînement et optimisation des modèles...\n")
    
    for model_name, config in models_config.items():
        print(f"  → {model_name}...")
        print(f"    GridSearchCV : {len(config['params'])} paramètres à tester")
        
        # GridSearchCV avec cross-validation
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=3,  # 3-fold CV
            scoring='average_precision',  # AUPRC (métrique clé pour déséquilibre)
            n_jobs=n_jobs,
            verbose=0
        )
        
        # Entraînement sur données SMOTE
        grid_search.fit(X_tr_balanced, y_tr_balanced)
        
        # Meilleur modèle
        best_model = grid_search.best_estimator_
        results['models'][model_name] = best_model
        results['best_params'][model_name] = grid_search.best_params_
        
        print(f"    ✓ Meilleurs paramètres : {grid_search.best_params_}")
        
       
        # ÉVALUATION SUR VALIDATION (SANS SMOTE!)
        
        # Prédictions
        y_val_pred = best_model.predict(X_val)
        y_val_proba = best_model.predict_proba(X_val)[:, 1]
        
        # Métriques
        auprc = average_precision_score(y_val, y_val_proba)
        auc_roc = roc_auc_score(y_val, y_val_proba)
        
        results['scores'][model_name] = {
            'AUPRC': auprc,
            'AUC-ROC': auc_roc
        }
        results['predictions'][model_name] = {
            'y_pred': y_val_pred,
            'y_proba': y_val_proba
        }
        
        print(f"     AUPRC : {auprc:.4f}")
        print(f"     AUC-ROC : {auc_roc:.4f}")
        
        # Classification report
        print(f"\n    Classification Report :")
        report = classification_report(y_val, y_val_pred, target_names=['Légitime', 'Fraude'])
        for line in report.split('\n'):
            if line.strip():
                print(f"      {line}")
        print()
    
    # ========================================
    # RÉSUMÉ COMPARATIF
    # ========================================
    print("\n" + "=" * 80)
    print("RÉSUMÉ DES PERFORMANCES (sur validation)")
    print("=" * 80)
    
    scores_df = pd.DataFrame(results['scores']).T
    scores_df = scores_df.sort_values('AUPRC', ascending=False)
    print(scores_df.to_string())
    
    best_model_name = scores_df.index[0]
    print(f"\n🏆 Meilleur modèle : {best_model_name} (AUPRC: {scores_df.loc[best_model_name, 'AUPRC']:.4f})")
    print("=" * 80)
    
    # Stocker les données de validation pour évaluation ultérieure
    results['validation_data'] = {
        'X_val': X_val,
        'y_val': y_val
    }
    
    return results


def train_anomaly_models(X_train, y_train,
                        feature_selection='auto',  # 'auto', 'top_k', ou liste
                        top_k=12,
                        contamination=0.005, 
                        n_neighbors_lof=50,   
                        random_state=42,
                        n_jobs=-1):
    """
    Entraîne des modèles de détection d'anomalies.
    
    Modèles : Isolation Forest, Local Outlier Factor
    
    Note : Ces modèles n'utilisent PAS SMOTE car ils apprennent
           ce qui est "normal" (pas besoin d'équilibrage).
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Features d'entraînement (déjà prétraitées)
    y_train : pd.Series
        Variable cible (utilisée uniquement pour évaluation)
    contamination : float, default=0.002
        Proportion attendue d'anomalies (0.172% ≈ 0.002)
    random_state : int, default=42
        Seed pour reproductibilité
    n_jobs : int, default=-1
        Nombre de CPU à utiliser
    
    Returns:
    --------
    results : dict
        Dictionnaire contenant les modèles entraînés et leurs performances
        {
            'models': {'IsolationForest': model, 'LOF': model},
            'scores': {...},
            'predictions': {...}
        }
    """
    
    print("\n" + "=" * 80)
    print("ENTRAÎNEMENT DES MODÈLES DE DÉTECTION D'ANOMALIES")
    print("=" * 80)
    
    results = {
        'models': {},
        'scores': {},
        'predictions': {}
    }

        # Feature selection si demandé
    if feature_selection == 'top_k':
        # Calculer importance avec RF rapide
        rf_selector = RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=n_jobs)
        rf_selector.fit(X_train, y_train)
        
        # Sélectionner top K
        importances = rf_selector.feature_importances_
        top_indices = importances.argsort()[-top_k:][::-1]
        selected_features = X_train.columns[top_indices].tolist()
        
        # Sauvegarder aussi anomaly_features si fourni
        anomaly_features_path = f"{save_dir}/anomaly_features.pkl"
        joblib.dump(selected_features, anomaly_features_path)

        
        X_train_reduced = X_train[selected_features]
        print(f"✓ Feature selection : {top_k} features sélectionnées")
        print(f"  Features : {selected_features}")
    elif isinstance(feature_selection, list):
        X_train_reduced = X_train[feature_selection]
        print(f"✓ Utilisation de {len(feature_selection)} features spécifiées")
    else:
        X_train_reduced = X_train
        print(f"✓ Utilisation de toutes les features ({X_train.shape[1]})")
    

   
    # 1. ISOLATION FOREST
    
    print(f"\n Isolation Forest...")
    print(f"  Contamination : {contamination}")
    
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_jobs=n_jobs,
        n_estimators=100,
        max_samples='auto',
        verbose=0
    )
    
    # Entraînement (utilise toutes les données, pas de SMOTE)
    iso_forest.fit(X_train_reduced)
    
    # Prédictions : -1 = anomalie, 1 = normal
    y_pred_iso = iso_forest.predict(X_train_reduced)
    y_pred_iso_binary = (y_pred_iso == -1).astype(int)  # Convertir en 0/1
    
    # Scores d'anomalie (plus négatif = plus anormal)
    y_scores_iso = iso_forest.score_samples(X_train_reduced)
    # Inverser pour avoir des scores positifs pour anomalies
    y_scores_iso_normalized = -y_scores_iso
    
    # Métriques
    auprc_iso = average_precision_score(y_train, y_scores_iso_normalized)
    auc_roc_iso = roc_auc_score(y_train, y_scores_iso_normalized)
    
    results['models']['IsolationForest'] = iso_forest
    results['scores']['IsolationForest'] = {
        'AUPRC': auprc_iso,
        'AUC-ROC': auc_roc_iso
    }
    results['predictions']['IsolationForest'] = {
        'y_pred': y_pred_iso_binary,
        'y_scores': y_scores_iso_normalized
    }
    
    print(f"   AUPRC : {auprc_iso:.4f}")
    print(f"   AUC-ROC : {auc_roc_iso:.4f}")
    
    # Classification report
    print(f"\n  Classification Report :")
    report_iso = classification_report(y_train, y_pred_iso_binary, 
                                       target_names=['Légitime', 'Fraude'])
    for line in report_iso.split('\n'):
        if line.strip():
            print(f"    {line}")
    
    
    # 2. LOCAL OUTLIER FACTOR
   
    print(f"\n Local Outlier Factor (LOF)...")
    print(f"  Contamination : {contamination}")
    
    lof = LocalOutlierFactor(
        contamination=contamination,
        n_neighbors=20,
        n_jobs=n_jobs,
        novelty=False  # Mode standard (pas novelty detection)
    )
    
    # LOF en mode non-novelty : fit_predict en une seule fois
    y_pred_lof = lof.fit_predict(X_train_reduced)
    y_pred_lof_binary = (y_pred_lof == -1).astype(int)
    
    # Scores d'anomalie (negative_outlier_factor_)
    y_scores_lof = -lof.negative_outlier_factor_  # Inverser pour cohérence
    
    # Métriques
    auprc_lof = average_precision_score(y_train, y_scores_lof)
    auc_roc_lof = roc_auc_score(y_train, y_scores_lof)
    
    results['models']['LOF'] = lof
    results['scores']['LOF'] = {
        'AUPRC': auprc_lof,
        'AUC-ROC': auc_roc_lof
    }
    results['predictions']['LOF'] = {
        'y_pred': y_pred_lof_binary,
        'y_scores': y_scores_lof
    }
    
    print(f"   AUPRC : {auprc_lof:.4f}")
    print(f"   AUC-ROC : {auc_roc_lof:.4f}")
    
    # Classification report
    print(f"\n  Classification Report :")
    report_lof = classification_report(y_train, y_pred_lof_binary,
                                       target_names=['Légitime', 'Fraude'])
    for line in report_lof.split('\n'):
        if line.strip():
            print(f"    {line}")
    
    
    # RÉSUMÉ COMPARATIF
    
    print("\n" + "=" * 80)
    print("RÉSUMÉ DES PERFORMANCES (Détection d'Anomalies)")
    print("=" * 80)
    
    scores_df = pd.DataFrame(results['scores']).T
    scores_df = scores_df.sort_values('AUPRC', ascending=False)
    print(scores_df.to_string())
    
    best_model_name = scores_df.index[0]
    print(f"\n Meilleur modèle : {best_model_name} (AUPRC: {scores_df.loc[best_model_name, 'AUPRC']:.4f})")
    print("=" * 80)
    
    return results


def save_models(supervised_results, anomaly_results, 
                scaler, selected_features,
                save_dir='models'):
    """
    Sauvegarde tous les modèles entraînés et objets nécessaires.

    Parameters:
    -----------
    supervised_results : dict
        Résultats de train_supervised_models()
    anomaly_results : dict
        Résultats de train_anomaly_models()
    scaler : RobustScaler
        Scaler utilisé pour Amount
    selected_features : list
        Liste des features utilisées
    save_dir : str, default='models'
        Dossier racine pour sauvegarder
    
    Returns:
    --------
    None
    """
    
    print("\n" + "=" * 80)
    print("SAUVEGARDE DES MODÈLES")
    print("=" * 80)
    
    # Créer l'arborescence
    os.makedirs(save_dir, exist_ok=True)
    #os.makedirs(f"{save_dir}/supervised", exist_ok=True)
    #os.makedirs(f"{save_dir}/anomaly", exist_ok=True)
    #os.makedirs(f"{save_dir}/preprocessing", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    saved_files = []
    
    
    # 1. MODÈLES SUPERVISÉS
   
    print("\n Sauvegarde des modèles supervisés...")
    
    model_mapping = {
        'RandomForest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    
    for model_name, filename in model_mapping.items():
        if model_name in supervised_results['models']:
            filepath = f"{save_dir}/{filename}"
            joblib.dump(supervised_results['models'][model_name], filepath)
            saved_files.append(filepath)
            print(f"  ✓ {model_name} → {filepath}")
    
    
    # 2. MODÈLES DE DÉTECTION D'ANOMALIES

    print("\n Sauvegarde des modèles d'anomalies...")
    
    anomaly_mapping = {
        'IsolationForest': 'isolation_forest.pkl',
        'LOF': 'lof.pkl'
    }
    
    for model_name, filename in anomaly_mapping.items():
        if model_name in anomaly_results['models']:
            filepath = f"{save_dir}/{filename}"
            joblib.dump(anomaly_results['models'][model_name], filepath)
            saved_files.append(filepath)
            print(f"  ✓ {model_name} → {filepath}")
    

    # 3. OBJETS DE PREPROCESSING
    
    print("\n Sauvegarde des objets de preprocessing...")
    
    # Scaler
    scaler_path = f"{save_dir}/scaler.pkl"
    joblib.dump(scaler, scaler_path)
    saved_files.append(scaler_path)
    
    # Features sélectionnées
    features_path = f"{save_dir}/selected_features.pkl"
    joblib.dump(selected_features, features_path)
    saved_files.append(features_path)
    
    # 4. MÉTADONNÉES
    
    print("\n Création du fichier de métadonnées...")
    
    metadata_path = f"{save_dir}/metadata.txt"
    
    with open(metadata_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODÈLES DE DÉTECTION DE FRAUDE - MÉTADONNÉES\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Date de création : {timestamp}\n\n")
        
        f.write("MODÈLES SUPERVISÉS\n")
        f.write("-" * 80 + "\n")
        for model_name in supervised_results['models'].keys():
            scores = supervised_results['scores'][model_name]
            params = supervised_results['best_params'][model_name]
            f.write(f"\n{model_name}:\n")
            f.write(f"  AUPRC : {scores['AUPRC']:.4f}\n")
            f.write(f"  AUC-ROC : {scores['AUC-ROC']:.4f}\n")
            f.write(f"  Meilleurs paramètres : {params}\n")
        
        f.write("\n\nMODÈLES DE DÉTECTION D'ANOMALIES\n")
        f.write("-" * 80 + "\n")
        for model_name in anomaly_results['models'].keys():
            scores = anomaly_results['scores'][model_name]
            f.write(f"\n{model_name}:\n")
            f.write(f"  AUPRC : {scores['AUPRC']:.4f}\n")
            f.write(f"  AUC-ROC : {scores['AUC-ROC']:.4f}\n")
        
        f.write("\n\nPREPROCESSING\n")
        f.write("-" * 80 + "\n")
        f.write(f"Nombre de features : {len(selected_features)}\n")
        f.write(f"Features : {selected_features}\n")
        
        f.write("\n\nFICHIERS SAUVEGARDÉS\n")
        f.write("-" * 80 + "\n")
        for filepath in saved_files:
            f.write(f"  - {filepath}\n")
    
    saved_files.append(metadata_path)
    



"""
==============================================================================
ÉVALUATION COMPLÈTE SUR TEST SET
==============================================================================
Fonction pour évaluer tous les modèles sauvegardés sur les données de test
et créer des visualisations comparatives complètes
==============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_auc_score, 
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
import warnings
warnings.filterwarnings('ignore')


def evaluate_models_on_test(X_test, y_test,
                            models_dir='models',
                            save_dir='images',
                            anomaly_features=None,
                            verbose=True):
    """
    Évalue tous les modèles sauvegardés sur le test set.
    
    Pipeline :
    1. Charge tous les modèles depuis models_dir
    2. Fait des prédictions sur X_test
    3. Calcule toutes les métriques importantes
    4. Crée des visualisations comparatives
    5. Génère un rapport détaillé
    
    Parameters:
    -----------
    X_test : pd.DataFrame
        Features de test (déjà prétraitées)
    y_test : pd.Series
        Labels de test
    models_dir : str, default='models'
        Dossier contenant les modèles sauvegardés
    save_dir : str, default='images'
        Dossier pour sauvegarder les visualisations
    verbose : bool, default=True
        Afficher les informations détaillées
    
    Returns:
    --------
    results : dict
        Dictionnaire contenant :
        {
            'metrics': DataFrame avec toutes les métriques par modèle,
            'predictions': dict avec prédictions de chaque modèle,
            'confusion_matrices': dict avec matrices de confusion,
            'figures': dict avec les figures matplotlib créées
        }
    """
    
    if verbose:
        print("=" * 80)
        print("ÉVALUATION DES MODÈLES SUR TEST SET")
        print("=" * 80)
        print(f"\nTest set : {len(X_test)} transactions")
        print(f"  - Fraudes : {y_test.sum()} ({y_test.mean()*100:.3f}%)")
        print(f"  - Légitimes : {len(y_test) - y_test.sum()}")
    
    # ========================================
    # 1. CHARGER TOUS LES MODÈLES
    # ========================================
    if verbose:
        print(f"\n[1/4] Chargement des modèles depuis '{models_dir}'...")
    
    models = {}
    model_files = {
        'XGBoost': f'{models_dir}/xgboost.pkl',
        'RandomForest': f'{models_dir}/random_forest.pkl',
        'IsolationForest': f'{models_dir}/isolation_forest.pkl',
        'LOF': f'{models_dir}/lof.pkl'
    }
    
    for model_name, filepath in model_files.items():
        if os.path.exists(filepath):
            models[model_name] = joblib.load(filepath)
            if verbose:
                print(f"  ✓ {model_name} chargé")
        else:
            if verbose:
                print(f"  ⚠️  {model_name} non trouvé ({filepath})")
    
    if len(models) == 0:
        raise FileNotFoundError(f"Aucun modèle trouvé dans {models_dir}")
    
    if verbose:
        print(f"\n✓ {len(models)} modèles chargés avec succès")
    
    if anomaly_features is None:
        anomaly_features_path = f'{models_dir}/anomaly_features.pkl'
        if os.path.exists(anomaly_features_path):
            anomaly_features = joblib.load(anomaly_features_path)
            if verbose:
                print(f"  ✓ Features anomalies auto-chargées : {len(anomaly_features)}")
    

    # ========================================
    # 2. PRÉDICTIONS ET CALCUL DES MÉTRIQUES
    # ========================================
    if verbose:
        print(f"\n[2/4] Calcul des prédictions et métriques...")
    
    results = {
        'metrics': {},
        'predictions': {},
        'probabilities': {},
        'confusion_matrices': {}
    }
    
    for model_name, model in models.items():
        if verbose:
            print(f"\n  → {model_name}...")
        
        # NOUVEAU : Sélectionner les features appropriées
        if model_name in ['IsolationForest', 'LOF']:
            if anomaly_features is None:
                if verbose:
                    print(f"    ⚠️  Pas de features spécifiées pour {model_name}, utilisation de toutes")
                X_model = X_test
            else:
                X_model = X_test[anomaly_features]  # ← FILTRAGE
                if verbose:
                    print(f"    ✓ Utilisation de {len(anomaly_features)} features sélectionnées")
        else:
            X_model = X_test  # Modèles supervisés utilisent toutes les features
        
        # Prédictions avec X_model au lieu de X_test
        if model_name in ['IsolationForest', 'LOF']:
            if model_name == 'IsolationForest':
                y_pred = model.predict(X_model)  # ← X_model !
                y_pred_binary = (y_pred == -1).astype(int)
                y_scores = -model.score_samples(X_model)  # ← X_model !
            else:  # LOF
                # (code LOF)
                continue
        else:
            # Modèles supervisés
            y_pred_binary = model.predict(X_model)
            y_scores = model.predict_proba(X_model)[:, 1]
        results['predictions'][model_name] = y_pred_binary
        results['probabilities'][model_name] = y_scores
        
        # Métriques
        precision = precision_score(y_test, y_pred_binary, zero_division=0)
        recall = recall_score(y_test, y_pred_binary, zero_division=0)
        f1 = f1_score(y_test, y_pred_binary, zero_division=0)
        auprc = average_precision_score(y_test, y_scores)
        auc_roc = roc_auc_score(y_test, y_scores)
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred_binary)
        results['confusion_matrices'][model_name] = cm
        
        # Stocker les métriques
        results['metrics'][model_name] = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'AUPRC': auprc,
            'AUC-ROC': auc_roc,
            'TP': cm[1, 1],
            'FP': cm[0, 1],
            'TN': cm[0, 0],
            'FN': cm[1, 0]
        }
        
        if verbose:
            print(f"    ✓ Precision: {precision:.4f}  |  Recall: {recall:.4f}  |  AUPRC: {auprc:.4f}")
    
    # Créer DataFrame des métriques
    metrics_df = pd.DataFrame(results['metrics']).T
    metrics_df = metrics_df.sort_values('AUPRC', ascending=False)
    results['metrics_df'] = metrics_df
    
    if verbose:
        print("\n" + "=" * 80)
        print("RÉSUMÉ DES MÉTRIQUES (TEST SET)")
        print("=" * 80)
        print(metrics_df[['Precision', 'Recall', 'F1-Score', 'AUPRC', 'AUC-ROC']].to_string())
    
    # ========================================
    # 3. CRÉER LES VISUALISATIONS
    # ========================================
    if verbose:
        print(f"\n[3/4] Création des visualisations...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    results['figures'] = {}
    
    # --- VISUALISATION 1 : Comparaison des métriques (barres) ---
    fig1 = _plot_metrics_comparison(metrics_df, save_dir, verbose)
    results['figures']['metrics_comparison'] = fig1
    
    # --- VISUALISATION 2 : Matrices de confusion ---
    fig2 = _plot_confusion_matrices(results['confusion_matrices'], save_dir, verbose)
    results['figures']['confusion_matrices'] = fig2
    
    # --- VISUALISATION 3 : Courbes ROC ---
    fig3 = _plot_roc_curves(y_test, results['probabilities'], save_dir, verbose)
    results['figures']['roc_curves'] = fig3
    
    # --- VISUALISATION 4 : Courbes Precision-Recall ---
    fig4 = _plot_pr_curves(y_test, results['probabilities'], save_dir, verbose)
    results['figures']['pr_curves'] = fig4
    
    # ========================================
    # 4. GÉNÉRER LE RAPPORT FINAL
    # ========================================
    if verbose:
        print(f"\n[4/4] Génération du rapport final...")
        _print_final_report(metrics_df, y_test)
    
    if verbose:
        print("\n" + "=" * 80)
        print("✓ ÉVALUATION TERMINÉE")
        print("=" * 80)
        print(f"\nFichiers générés dans '{save_dir}/' :")
        print("  1. 08_test_metrics_comparison.png")
        print("  2. 09_test_confusion_matrices.png")
        print("  3. 10_test_roc_curves.png")
        print("  4. 11_test_precision_recall_curves.png")
    
    return results


# ========================================
# FONCTIONS AUXILIAIRES DE VISUALISATION
# ========================================

def _plot_metrics_comparison(metrics_df, save_dir, verbose):
    """Graphique comparatif des 5 métriques principales."""
    
    if verbose:
        print("  → Création du graphique de comparaison des métriques...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Comparaison des Performances sur Test Set', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    metrics_to_plot = ['AUPRC', 'AUC-ROC', 'Precision', 'Recall', 'F1-Score']
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for idx, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        ax = axes[idx // 3, idx % 3]
        
        sorted_df = metrics_df.sort_values(metric, ascending=True)
        bars = ax.barh(range(len(sorted_df)), sorted_df[metric].values,
                      color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df.index, fontsize=11, fontweight='bold')
        ax.set_xlabel('Score', fontsize=12)
        ax.set_title(metric, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3)
        
        # Annotations
        for i, (bar, value) in enumerate(zip(bars, sorted_df[metric].values)):
            ax.text(value + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', va='center', fontsize=10, fontweight='bold')
    
    # Supprimer le subplot vide (6ème)
    fig.delaxes(axes[1, 2])
    
    # Ajouter un résumé textuel dans le subplot vide
    ax_summary = axes[1, 2]
    fig.add_subplot(ax_summary)
    
    best_model = metrics_df.index[0]
    best_auprc = metrics_df['AUPRC'].iloc[0]
    
    summary_text = f"""
    🏆 MEILLEUR MODÈLE
    
    Modèle : {best_model}
    AUPRC : {best_auprc:.4f}
    
    Métriques :
    • Precision : {metrics_df.loc[best_model, 'Precision']:.3f}
    • Recall : {metrics_df.loc[best_model, 'Recall']:.3f}
    • F1-Score : {metrics_df.loc[best_model, 'F1-Score']:.3f}
    
    Détection :
    • TP : {int(metrics_df.loc[best_model, 'TP'])}
    • FP : {int(metrics_df.loc[best_model, 'FP'])}
    • FN : {int(metrics_df.loc[best_model, 'FN'])}
    """
    
    ax_summary.text(0.1, 0.5, summary_text, fontsize=11, 
                   verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_summary.axis('off')
    
    plt.tight_layout()
    filepath = f'{save_dir}/08_test_metrics_comparison.png'
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    
    if verbose:
        print(f"    ✓ Sauvegardé : {filepath}")
    
    return fig


def _plot_confusion_matrices(confusion_matrices, save_dir, verbose):
    """Matrices de confusion pour tous les modèles."""
    
    if verbose:
        print("  → Création des matrices de confusion...")
    
    n_models = len(confusion_matrices)
    cols = 2
    rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5*rows))
    fig.suptitle('Matrices de Confusion sur Test Set', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        
        # Normaliser par ligne (pour voir les pourcentages par classe)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   cbar=True, ax=ax, square=True,
                   xticklabels=['Légitime', 'Fraude'],
                   yticklabels=['Légitime', 'Fraude'])
        
        ax.set_title(f'{model_name}\n(Valeurs normalisées par ligne)', 
                    fontsize=13, fontweight='bold', pad=10)
        ax.set_ylabel('Vraie Classe', fontsize=11)
        ax.set_xlabel('Classe Prédite', fontsize=11)
        
        # Ajouter les valeurs absolues en annotation
        for i in range(2):
            for j in range(2):
                text = ax.text(j + 0.5, i + 0.7, f'({int(cm[i, j])})',
                             ha="center", va="center", fontsize=9,
                             color="gray")
    
    # Masquer les subplots vides
    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    filepath = f'{save_dir}/09_test_confusion_matrices.png'
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    
    if verbose:
        print(f"    ✓ Sauvegardé : {filepath}")
    
    return fig


def _plot_roc_curves(y_test, probabilities, save_dir, verbose):
    """Courbes ROC pour tous les modèles."""
    
    if verbose:
        print("  → Création des courbes ROC...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'XGBoost': '#e74c3c', 'RandomForest': '#3498db', 
              'IsolationForest': '#f39c12', 'LOF': '#95a5a6'}
    
    for model_name, y_scores in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        auc = roc_auc_score(y_test, y_scores)
        
        ax.plot(fpr, tpr, linewidth=3, 
               label=f'{model_name} (AUC = {auc:.4f})',
               color=colors.get(model_name, '#2ecc71'))
    
    # Ligne diagonale (modèle aléatoire)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Aléatoire (AUC = 0.50)')
    
    ax.set_xlabel('Taux de Faux Positifs (FPR)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Taux de Vrais Positifs (TPR / Recall)', fontsize=13, fontweight='bold')
    ax.set_title('Courbes ROC - Test Set', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    filepath = f'{save_dir}/10_test_roc_curves.png'
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    
    if verbose:
        print(f"    ✓ Sauvegardé : {filepath}")
    
    return fig


def _plot_pr_curves(y_test, probabilities, save_dir, verbose):
    """Courbes Precision-Recall pour tous les modèles."""
    
    if verbose:
        print("  → Création des courbes Precision-Recall...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'XGBoost': '#e74c3c', 'RandomForest': '#3498db',
              'IsolationForest': '#f39c12', 'LOF': '#95a5a6'}
    
    baseline = y_test.sum() / len(y_test)
    
    for model_name, y_scores in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        auprc = average_precision_score(y_test, y_scores)
        
        ax.plot(recall, precision, linewidth=3,
               label=f'{model_name} (AUPRC = {auprc:.4f})',
               color=colors.get(model_name, '#2ecc71'))
    
    # Baseline (taux de fraude)
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=2,
              label=f'Baseline (taux fraude = {baseline:.4f})')
    
    ax.set_xlabel('Recall (Taux de Détection)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Precision (Fiabilité des Alertes)', fontsize=13, fontweight='bold')
    ax.set_title('Courbes Precision-Recall - Test Set', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='best', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    filepath = f'{save_dir}/11_test_precision_recall_curves.png'
    fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    
    if verbose:
        print(f"    ✓ Sauvegardé : {filepath}")
    
    return fig


def _print_final_report(metrics_df, y_test):
    """Imprime un rapport final détaillé."""
    
    print("\n" + "=" * 80)
    print("RAPPORT FINAL - ÉVALUATION TEST SET")
    print("=" * 80)
    
    best_model = metrics_df.index[0]
    best_metrics = metrics_df.loc[best_model]
    
    print(f"\n🏆 MEILLEUR MODÈLE : {best_model}")
    print("-" * 80)
    print(f"  AUPRC (métrique principale) : {best_metrics['AUPRC']:.4f}")
    print(f"  AUC-ROC : {best_metrics['AUC-ROC']:.4f}")
    print(f"  Precision : {best_metrics['Precision']:.4f} ({best_metrics['Precision']*100:.1f}%)")
    print(f"  Recall : {best_metrics['Recall']:.4f} ({best_metrics['Recall']*100:.1f}%)")
    print(f"  F1-Score : {best_metrics['F1-Score']:.4f}")
    
    print(f"\n📊 DÉTECTION DES FRAUDES :")
    total_frauds = y_test.sum()
    detected = int(best_metrics['TP'])
    missed = int(best_metrics['FN'])
    
    print(f"  Total de fraudes dans test : {total_frauds}")
    print(f"  Fraudes détectées (TP) : {detected} ({detected/total_frauds*100:.1f}%)")
    print(f"  Fraudes manquées (FN) : {missed} ({missed/total_frauds*100:.1f}%)")
    
    print(f"\n⚠️  FAUX POSITIFS :")
    fp = int(best_metrics['FP'])
    total_legit = len(y_test) - total_frauds
    print(f"  Faux positifs (FP) : {fp}")
    print(f"  Taux de FP : {fp/total_legit*100:.3f}% des transactions légitimes")
    
    print(f"\n💰 IMPACT BUSINESS (Estimation) :")
    print(f"  Si montant moyen fraude = 150€ :")
    print(f"    - Pertes évitées : {detected * 150:,}€")
    print(f"    - Pertes résiduelles : {missed * 150:,}€")
    print(f"    - Coût faux positifs (×12.50€) : {fp * 12.5:,}€")
    
    net_savings = (detected * 150) - (fp * 12.5)
    print(f"    - 💎 Économies nettes : {net_savings:,}€")
    
    print("\n" + "=" * 80)
