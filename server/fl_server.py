import os
import numpy as np
import flwr as fl
from supabase import create_client, Client
import base64
import json
import socket
from urllib.parse import urlparse

SUPABASE_URL = os.environ.get("VITE_SUPABASE_URL") or os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("VITE_SUPABASE_ANON_KEY") or os.environ.get("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

_run_id = os.environ.get("FL_RUN_ID", "")
_num_rounds = int(os.environ.get("FL_NUM_ROUNDS", "3"))
_model_type = os.environ.get("FL_MODEL_TYPE", "logreg")


def _sb() -> Client:
    url = SUPABASE_URL
    key = SUPABASE_SERVICE_KEY or SUPABASE_KEY
    if not url or not key:
        raise RuntimeError(
            "Supabase env vars missing. Set SUPABASE_URL and SUPABASE_SERVICE_KEY (recommended) "
            "or SUPABASE_ANON_KEY."
        )
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise RuntimeError(
            f"Invalid SUPABASE_URL '{url}'. Expected like 'https://<project-ref>.supabase.co'."
        )
    hostname = parsed.hostname or ""

    # Key-ref vs URL mismatch check (helps catch typos in project ref).
    try:
        parts = key.split(".")
        if len(parts) >= 2:
            payload = parts[1]
            payload += "=" * ((4 - (len(payload) % 4)) % 4)
            claims = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
            ref = claims.get("ref")
            if (
                isinstance(ref, str)
                and ref
                and hostname.endswith(".supabase.co")
                and not hostname.startswith(ref + ".")
            ):
                raise RuntimeError(
                    f"SUPABASE_URL host '{hostname}' does not match project ref '{ref}' encoded in the Supabase key. "
                    f"Expected SUPABASE_URL like 'https://{ref}.supabase.co'."
                )
    except RuntimeError:
        raise
    except Exception:
        pass

    try:
        socket.getaddrinfo(hostname, 443)
    except OSError as e:
        raise RuntimeError(
            f"Cannot resolve SUPABASE_URL host '{hostname}'. DNS error: {e}"
        ) from e
    return create_client(url, key)


def _insert_round_metric(round_idx: int, global_acc, grad_norm):
    _sb().table("fl_round_metrics").insert({
        "run_id": _run_id,
        "round": round_idx,
        "global_accuracy": global_acc,
        "aggregated_grad_norm": grad_norm,
    }).execute()


def _finalize_run(status: str) -> None:
    _sb().table("fl_runs").update({
        "status": status,
    }).eq("id", _run_id).execute()


class Strategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round: int, results, failures):
        agg = super().aggregate_fit(server_round, results, failures)
        grad_norm = None
        try:
            deltas = []
            for _, fit_res in results:
                gn = fit_res.metrics.get("grad_norm") if fit_res.metrics else None
                if gn is not None:
                    deltas.append(gn)
            if deltas:
                grad_norm = float(np.mean(deltas))
        except Exception:
            pass
        _insert_round_metric(server_round, global_acc=None, grad_norm=grad_norm)
        return agg

    def aggregate_evaluate(self, server_round: int, results, failures):
        agg = super().aggregate_evaluate(server_round, results, failures)
        global_acc = None
        try:
            if agg and agg[1] is not None:
                metrics = agg[1] or {}
                acc = metrics.get("accuracy")
                if acc is not None:
                    global_acc = float(acc)
        except Exception:
            pass
        _insert_round_metric(server_round, global_acc=global_acc, grad_norm=None)
        return agg


def start_server(server_address: str, stop_event) -> None:
    _sb().table("fl_runs").update({"status": "running"}).eq("id", _run_id).execute()
    strategy = Strategy(min_fit_clients=2, min_available_clients=2, min_evaluate_clients=2)
    fl.server.start_server(server_address=server_address, strategy=strategy, config=fl.server.ServerConfig(num_rounds=_num_rounds))
    _finalize_run("completed")
