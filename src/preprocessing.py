import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler



def data_split():

    df = pd.read_csv('../data/creditcard.csv')
    # Split 80% train / 20% test avec stratification
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  # Maintient le ratio de classes
    )
    return  X_train, X_test, y_train, y_test


def create_temporal_features(df, time_col='Time', verbose=True):
    """
    Crée des features temporelles avancées à partir de la colonne Time.
    
    Configuration B Complète :
    - Hour_sin, Hour_cos : Encodage cyclique de l'heure
    - Is_Night, Is_Morning, Is_Afternoon, Is_Evening : Périodes de la journée
    - Day : Jour depuis le début (0-2)
    - Transactions_per_hour : Densité de transactions par heure
    - Time_normalized : Time normalisé entre 0 et 1
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame contenant la colonne Time
    time_col : str
        Nom de la colonne temporelle (défaut: 'Time')
    verbose : bool
        Afficher les informations de création (défaut: True)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame avec les nouvelles features temporelles ajoutées
    """
    
    if verbose:
        print("\n Création des features temporelles...")
        print(f"Colonne source : {time_col}")
        print(f"Min : {df[time_col].min()}, Max : {df[time_col].max()}")
    
    # Créer une copie pour ne pas modifier l'original
    df_new = df.copy()
    
    # ========================================
    # 1. HEURE DE LA JOURNÉE (0-23)
    # ========================================
    df_new['Hour'] = (df_new[time_col] / 3600) % 24
    if verbose:
        print(f"\n✓ Hour créée (range: {df_new['Hour'].min():.2f} - {df_new['Hour'].max():.2f})")
    
    # ========================================
    # 2. ENCODAGE CYCLIQUE (SIN/COS)
    # ========================================
    # Convertir l'heure en coordonnées sur un cercle
    df_new['Hour_sin'] = np.sin(2 * np.pi * df_new['Hour'] / 24)
    df_new['Hour_cos'] = np.cos(2 * np.pi * df_new['Hour'] / 24)
    
    if verbose:
        print(f"✓ Hour_sin créée (range: {df_new['Hour_sin'].min():.3f} - {df_new['Hour_sin'].max():.3f})")
        print(f"✓ Hour_cos créée (range: {df_new['Hour_cos'].min():.3f} - {df_new['Hour_cos'].max():.3f})")
    
    # ========================================
    # 3. PÉRIODES DE LA JOURNÉE (CATÉGORIELLES)
    # ========================================
    # Nuit : 22h-6h (heures à haut risque de fraude)
    df_new['Is_Night'] = ((df_new['Hour'] >= 22) | (df_new['Hour'] < 6)).astype(int)
    
    # Matin : 6h-12h
    df_new['Is_Morning'] = ((df_new['Hour'] >= 6) & (df_new['Hour'] < 12)).astype(int)
    
    # Après-midi : 12h-18h
    df_new['Is_Afternoon'] = ((df_new['Hour'] >= 12) & (df_new['Hour'] < 18)).astype(int)
    
    # Soir : 18h-22h
    df_new['Is_Evening'] = ((df_new['Hour'] >= 18) & (df_new['Hour'] < 22)).astype(int)
    
    if verbose:
        print(f"\n✓ Périodes catégorielles créées :")
        print(f"  - Is_Night     : {df_new['Is_Night'].sum()} transactions ({df_new['Is_Night'].mean()*100:.1f}%)")
        print(f"  - Is_Morning   : {df_new['Is_Morning'].sum()} transactions ({df_new['Is_Morning'].mean()*100:.1f}%)")
        print(f"  - Is_Afternoon : {df_new['Is_Afternoon'].sum()} transactions ({df_new['Is_Afternoon'].mean()*100:.1f}%)")
        print(f"  - Is_Evening   : {df_new['Is_Evening'].sum()} transactions ({df_new['Is_Evening'].mean()*100:.1f}%)")
    
    # ========================================
    # 4. JOUR DEPUIS LE DÉBUT
    # ========================================
    df_new['Day'] = df_new[time_col] / (3600 * 24)
    
    if verbose:
        print(f"\n✓ Day créée (range: {df_new['Day'].min():.2f} - {df_new['Day'].max():.2f})")
    
    # ========================================
    # 5. DENSITÉ DE TRANSACTIONS PAR HEURE
    # ========================================
    # Calculer combien de transactions ont lieu à chaque heure
    hour_counts = df_new['Hour'].round().value_counts()
    df_new['Transactions_per_hour'] = df_new['Hour'].round().map(hour_counts)
    
    if verbose:
        print(f"\n✓ Transactions_per_hour créée")
        print(f"  Min : {df_new['Transactions_per_hour'].min()}")
        print(f"  Max : {df_new['Transactions_per_hour'].max()}")
        print(f"  Mean: {df_new['Transactions_per_hour'].mean():.0f}")
    
    # ========================================
    # 6. TIME NORMALISÉ (0-1)
    # ========================================
    df_new['Time_normalized'] = df_new[time_col] / df_new[time_col].max()
    
    if verbose:
        print(f"\n✓ Time_normalized créée (range: {df_new['Time_normalized'].min():.4f} - {df_new['Time_normalized'].max():.4f})")
    
    # ========================================
    # RÉSUMÉ DES FEATURES CRÉÉES
    # ========================================
    new_features = [
        'Hour', 'Hour_sin', 'Hour_cos',
        'Is_Night', 'Is_Morning', 'Is_Afternoon', 'Is_Evening',
        'Day', 'Transactions_per_hour', 'Time_normalized'
    ]
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"✓ FEATURES TEMPORELLES CRÉÉES : {len(new_features)} features")
        print("=" * 80)
        for i, feat in enumerate(new_features, 1):
            print(f"  {i:2}. {feat}")
    
    return df_new, new_features

def data_preprocessing(X, y, temporal_features_selected ,test_data=False, scaler=None):
    """
    Prétraite les données pour la modélisation.
    
    Étapes :
    1. Standardisation de 'Amount' avec RobustScaler
    2. Création des features temporelles
    3. Sélection des features temporelles importantes (>= 0.1)
    4. Suppression de 'Time' original
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features d'entrée (doit contenir 'Time' et 'Amount')
    y : pd.Series
        Variable cible (Class)
    test_data : bool, default=False
        True si données de test (utilise scaler pré-entraîné)
        False si données d'entraînement (entraîne nouveau scaler)
    scaler : RobustScaler, default=None
        Scaler pré-entraîné (obligatoire si test_data=True)
    
    Returns:
    --------
    X_processed : pd.DataFrame
        Features prétraitées
    y : pd.Series
        Variable cible (inchangée)
    scaler : RobustScaler
        Scaler entraîné (à réutiliser pour test set)
    """
    
    # Vérifications
    if test_data and scaler is None:
        raise ValueError("Scaler requis pour test_data=True")
    
    if 'Time' not in X.columns or 'Amount' not in X.columns:
        raise ValueError("Colonnes 'Time' et 'Amount' requises dans X")
    
    # Copie des données
    X_processed = X.copy()
    

    # 1. STANDARDISATION DE AMOUNT

    if test_data:
        X_processed['Amount'] = scaler.transform(X_processed[['Amount']])
    else:
        scaler = RobustScaler()
        X_processed['Amount'] = scaler.fit_transform(X_processed[['Amount']])

    # 2. CRÉATION DES FEATURES TEMPORELLES

    temp_df = X_processed.copy()
    temp_df['Class'] = y
    
    temp_df_enhanced, all_temporal_features = create_temporal_features(
        temp_df, 
        time_col='Time', 
        verbose=False
    )
    
    X_processed = temp_df_enhanced.drop('Class', axis=1)
    

    # 3. SÉLECTION DES FEATURES TEMPORELLES IMPORTANTES

    # Garder V1-V28 + Amount + features temporelles sélectionnées
    pca_features = [f'V{i}' for i in range(1, 29)]
    features_to_keep = pca_features + ['Amount'] + temporal_features_selected
    
    # Vérifier que toutes les features existent
    missing_features = [f for f in features_to_keep if f not in X_processed.columns]
    if missing_features:
        raise ValueError(f"Features manquantes : {missing_features}")
    
    X_processed = X_processed[features_to_keep]
    



    data_type = "TEST" if test_data else "TRAIN"
    print(f"✓ Preprocessing {data_type} : {X_processed.shape[0]} transactions, "
          f"{X_processed.shape[1]} features")
    
    return X_processed, y, scaler


    





