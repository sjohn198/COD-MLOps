import json
import os
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

def create_global_scaler(data_files):
    scaler = StandardScaler()
    feature_cols = ["balls", "inning_topbot", "strikes", "on_3b", "on_2b", "on_1b", "outs_when_up", "inning", "run_diff"]

    for f in data_files:
        df = pd.read_parquet(f)
        chunk = df[feature_cols].astype(float)
        scaler.partial_fit(chunk)
    
    joblib.dump(scaler, "./data/baseball_scaler.pkl")



if __name__ == "__main__":
    print("running build_scaler")
    train_files = []
    validate_files = []
    test_files = []

    if os.path.exists("train_files.json") and os.path.exists("validate_files.json") and os.path.exists("test_files.json"):
        train_files = json.load("train_files.json")
        validate_files = json.load("validate_files.json")
        test_files = json.load("test_files.json")
    else:
        for i, (root, dirs, files) in enumerate(os.walk("./dataset")):
            #sort by year to avoid temporal leakeage
            for f in files:
                if "year=2024" in root:
                    validate_files.append(root + "/" + f)
                elif "year=2025" in root:
                    test_files.append(root + "/" + f)
                else:
                    train_files.append(root + "/" + f)
        with open("test_files.json", "w") as f1:
            json.dump(test_files, f1)
        with open("train_files.json", "w") as f2:
            json.dump(train_files, f2)
        with open("validate_files.json", "w") as f3:
            json.dump(validate_files, f3)

    scaler_path = "./data/baseball_scaler.pkl"
    if not os.path.exists(scaler_path):
        scaler = create_global_scaler(train_files)

    print("sclaer built")

    