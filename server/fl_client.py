import argparse
import os
from typing import Tuple
import numpy as np
import pandas as pd
import flwr as fl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from supabase import create_client

SUPABASE_URL = os.environ.get("VITE_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("VITE_SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_ANON_KEY")
RUN_ID = os.environ.get("FL_RUN_ID", "")


def sb():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def load_dataset(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    X = df.drop(columns=["target"]).to_numpy(dtype=float)
    y = df["target"].to_numpy(dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, dataset_path: str, label: str):
        self.label = label
        self.round = 0
        self.X_train, self.X_test, self.y_train, self.y_test = load_dataset(dataset_path)
        self.model = LogisticRegression(max_iter=200)
        if self.X_train.shape[0] > 2:
            self.model.fit(self.X_train[:2], self.y_train[:2])

    def get_parameters(self, config):
        coef = self.model.coef_ if hasattr(self.model, "coef_") else np.zeros((1, self.X_train.shape[1]))
        intercept = self.model.intercept_ if hasattr(self.model, "intercept_") else np.zeros((1,))
        return [coef.astype(np.float64), intercept.astype(np.float64)]

    def set_parameters(self, parameters):
        coef, intercept = parameters
        self.model.coef_ = np.array(coef)
        self.model.intercept_ = np.array(intercept)
        self.model.classes_ = np.array([0, 1])

    def fit(self, parameters, config):
        self.round += 1
        self.set_parameters(parameters)
        prev_params = [p.copy() for p in parameters]
        self.model.fit(self.X_train, self.y_train)
        new_params = self.get_parameters({})
        deltas = [np.linalg.norm((n - p).ravel(), ord=2) for p, n in zip(prev_params, new_params)]
        grad_norm = float(np.mean(deltas))
        num_examples = int(self.X_train.shape[0])
        # Insert client update (grad norm) for this round
        try:
            client = sb()
            if client and RUN_ID:
                client.table("fl_client_updates").insert({
                    "run_id": RUN_ID,
                    "round": self.round,
                    "client_user_id": None,
                    "client_label": self.label,
                    "local_accuracy": None,
                    "grad_norm": grad_norm,
                    "num_examples": num_examples,
                }).execute()
        except Exception:
            pass
        return new_params, num_examples, {"grad_norm": grad_norm}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        preds = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        loss = float(1.0 - acc)
        # Insert client local accuracy for this round
        try:
            client = sb()
            if client and RUN_ID:
                client.table("fl_client_updates").insert({
                    "run_id": RUN_ID,
                    "round": self.round,
                    "client_user_id": None,
                    "client_label": self.label,
                    "local_accuracy": float(acc),
                    "grad_norm": None,
                    "num_examples": int(self.X_test.shape[0]),
                }).execute()
        except Exception:
            pass
        return loss, int(self.X_test.shape[0]), {"accuracy": float(acc)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    dataset_path = os.path.join(os.path.dirname(__file__), "data", args.dataset)
    label = os.path.splitext(os.path.basename(dataset_path))[0]
    client = FlowerClient(dataset_path, label)
    fl.client.start_numpy_client(server_address=args.server, client=client)


if __name__ == "__main__":
    main()
