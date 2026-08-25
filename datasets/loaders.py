"""
Unified dataset loaders for NSL-KDD, UNSW-NB15, and CIC-IDS-2017.
Returns (X_train, y_train, X_test, y_test, feature_names).
"""

from pathlib import Path
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "database"

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "attack_type", "difficulty_level",
]

ATTACK_MAPPING = {
    "normal": "normal",
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "mailbomb": "dos",
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "satan": "probe", "mscan": "probe", "saint": "probe",
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l",
    "multihop": "r2l", "phf": "r2l", "spy": "r2l", "warezclient": "r2l",
    "warezmaster": "r2l", "sendmail": "r2l", "named": "r2l",
    "snmpgetattack": "r2l", "snmpguess": "r2l", "xlock": "r2l",
    "xsnoop": "r2l", "worm": "r2l",
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "httptunnel": "u2r", "ps": "u2r", "sqlattack": "u2r",
    "xterm": "u2r",
}


def list_datasets():
    return ["nsl-kdd", "unsw-nb15", "cic-ids-2017"]


def _encode_categoricals(train_df, test_df, cat_cols):
    """Label-encode categorical columns consistently across train/test."""
    encoders = {}
    for col in cat_cols:
        if col not in train_df.columns:
            continue
        le = LabelEncoder()
        combined = pd.concat([train_df[col].astype(str), test_df[col].astype(str)])
        le.fit(combined)
        train_df[col] = le.transform(train_df[col].astype(str))
        test_df[col] = le.transform(test_df[col].astype(str))
        encoders[col] = le
    return encoders


def _to_binary_labels(series):
    """Convert attack labels to binary (0=normal, 1=attack)."""
    s = series.astype(str).str.strip().str.lower()
    if s.isin(["0", "1"]).all():
        return s.astype(int)
    return (s != "normal").astype(int)


def load_nsl_kdd(scale=True):
    train_path = DB / "KDDTrain+.txt"
    test_path = DB / "KDDTest+.txt"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"NSL-KDD not found in {DB}. Run: python -m datasets.download --dataset nsl-kdd"
        )

    train_df = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    test_df = pd.read_csv(test_path, header=None, names=NSL_KDD_COLUMNS)

    train_df["attack_category"] = train_df["attack_type"].map(ATTACK_MAPPING).fillna("unknown")
    test_df["attack_category"] = test_df["attack_type"].map(ATTACK_MAPPING).fillna("unknown")

    y_train = (train_df["attack_category"] != "normal").astype(int)
    y_test = (test_df["attack_category"] != "normal").astype(int)

    drop_cols = ["attack_type", "difficulty_level", "attack_category"]
    cat_cols = ["protocol_type", "service", "flag"]
    _encode_categoricals(train_df, test_df, cat_cols)

    X_train = train_df.drop(columns=drop_cols).values.astype(float)
    X_test = test_df.drop(columns=drop_cols).values.astype(float)
    feature_names = [c for c in train_df.columns if c not in drop_cols]

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train.values, X_test, y_test.values, feature_names


def load_unsw_nb15(scale=True, test_size=0.2):
    train_path = DB / "UNSW_NB15_training-set.csv"
    test_path = DB / "UNSW_NB15_testing-set.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"UNSW-NB15 not found in {DB}. Run: python -m datasets.download --dataset unsw-nb15"
        )

    train_df = pd.read_csv(train_path)
    use_official_test = test_path.exists() and test_path.stat().st_size > 25_000_000

    if use_official_test:
        test_df = pd.read_csv(test_path)
    else:
        from sklearn.model_selection import train_test_split
        label_col = "label" if "label" in train_df.columns else "Label"
        train_df, test_df = train_test_split(
            train_df, test_size=test_size, random_state=42, stratify=train_df[label_col]
        )
        print(f"  [UNSW-NB15] Using train/test split (official test set unavailable)")

    label_col = "label" if "label" in train_df.columns else "Label"
    y_train = train_df[label_col].astype(int)
    y_test = test_df[label_col].astype(int)

    drop_cols = [label_col, "id", "attack_cat", "Attack_cat"]
    drop_cols = [c for c in drop_cols if c in train_df.columns]

    cat_cols = [c for c in train_df.columns if train_df[c].dtype == object]
    cat_cols = [c for c in cat_cols if c not in drop_cols]
    _encode_categoricals(train_df, test_df, cat_cols)

    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    X_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    X_test = test_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train.values, X_test, y_test.values, feature_cols


def load_cic_ids_2017(scale=True, max_rows_per_file=150_000):
    """
    Load CIC-IDS-2017 CSV files. Large dataset — subsample per file by default.
    """
    cic_dir = DB / "CIC-IDS-2017"
    patterns = [
        str(cic_dir / "**" / "MachineLearningCVE" / "*.csv"),
        str(cic_dir / "**" / "*.csv"),
        str(DB / "MachineLearningCVE" / "*.csv"),
    ]
    files = []
    for pat in patterns:
        files = glob.glob(pat, recursive=True)
        if files:
            break

    if not files:
        raise FileNotFoundError(
            f"CIC-IDS-2017 not found. Run: python -m datasets.download --dataset cic-ids-2017"
        )

    frames = []
    for fpath in sorted(files):
        df = pd.read_csv(fpath, encoding="latin-1", low_memory=False)
        df.columns = df.columns.str.strip()
        if len(df) > max_rows_per_file:
            df = df.sample(n=max_rows_per_file, random_state=42)
        frames.append(df)

    full = pd.concat(frames, ignore_index=True)
    full.replace([np.inf, -np.inf], np.nan, inplace=True)

    label_col = "Label" if "Label" in full.columns else "label"
    full[label_col] = full[label_col].astype(str).str.strip()
    y = _to_binary_labels(full[label_col])

    drop_cols = [label_col, "Flow ID", "Src IP", "Dst IP", "Timestamp"]
    drop_cols = [c for c in drop_cols if c in full.columns]

    X_df = full.drop(columns=drop_cols)
    cat_cols = [c for c in X_df.columns if X_df[c].dtype == object]
    dummy_train = X_df.copy()
    dummy_test = X_df.copy()
    _encode_categoricals(dummy_train, dummy_test, cat_cols)

    X = X_df.apply(pd.to_numeric, errors="coerce").fillna(0).values
    feature_names = list(X_df.columns)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y.values, test_size=0.2, random_state=42, stratify=y
    )

    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, feature_names


def load_dataset(name: str, scale=True, **kwargs):
    name = name.lower().replace("_", "-")
    loaders = {
        "nsl-kdd": load_nsl_kdd,
        "unsw-nb15": load_unsw_nb15,
        "cic-ids-2017": load_cic_ids_2017,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(loaders)}")
    return loaders[name](scale=scale, **kwargs)
