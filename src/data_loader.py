import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess():
    Categoricical_features = [
    'workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country'
]
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                    'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

    df = pd.read_csv(url, names=column_names, skipinitialspace=True)



    for col in Categoricical_features:
        df[col] = df[col].replace('?', 'Unknown')

    X = df[Categoricical_features].copy()
    y = (df['income'] == '>50K').astype(int)

    # Encoding
    features_after_encoding = {}
    for col in Categoricical_features:
        features_after_encoding[col] = LabelEncoder()
        X[col] = features_after_encoding[col].fit_transform(X[col])

    X = X.values

    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {Counter(y)}")

    # Split 70/15/15
    X_train_nb, X_temp_nb, y_train_nb, y_temp_nb = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val_nb, X_test_nb, y_val_nb, y_test_nb = train_test_split(
        X_temp_nb, y_temp_nb, test_size=0.5, stratify=y_temp_nb, random_state=42
    )

    print(f"Train:{len(y_train_nb)}, Val:{len(y_val_nb)}, Test:{len(y_test_nb)}")

    return X_train_nb, X_val_nb, X_test_nb, y_train_nb, y_val_nb, y_test_nb, features_after_encoding
