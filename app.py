"""
Network Intrusion Detection System — Flask Server
Real-time KDD feature extraction from live HTTP connections
"""

import time
import uuid
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from collections import defaultdict, deque
from flask import Flask, request, jsonify, g
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

FEATURE_NAMES = [
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
]


# ─────────────────────────────────────────────
#  Real-time connection tracker
#  Computes KDD-style features from live traffic
# ─────────────────────────────────────────────

class ConnectionTracker:
    """
    Tracks the last 2 seconds of connections per IP.
    Mirrors the KDD time-window features used in NSL-KDD training.
    """

    WINDOW = 2.0  # seconds — matches KDD dataset time window

    def __init__(self):
        # ip -> deque of (timestamp, path, method, status_code, bytes_sent)
        self._log: dict[str, deque] = defaultdict(lambda: deque())
        self._lock = __import__('threading').Lock()

    def record(self, ip: str, path: str, method: str,
               content_len: int, status_code: int = 0):
        now = time.time()
        with self._lock:
            dq = self._log[ip]
            dq.append((now, path, method, content_len, status_code))
            # prune old entries
            cutoff = now - self.WINDOW
            while dq and dq[0][0] < cutoff:
                dq.popleft()

    def get_kdd_features(self, ip: str, path: str, method: str,
                         content_len: int) -> dict:
        """
        Return a dict of KDD time-window features computed from
        real observed traffic in the last 2 seconds.
        """
        now = time.time()
        cutoff = now - self.WINDOW

        with self._lock:
            dq = self._log[ip]
            window = [(ts, p, m, cl) for ts, p, m, cl, _ in dq if ts >= cutoff]

        count      = len(window)                         # connections in window
        srv_count  = sum(1 for _, p, _, _ in window if p == path)
        src_bytes  = content_len
        dst_bytes  = 0                                   # unknown at request time

        # error rate proxy: repeated rapid short requests (flood pattern)
        serror_rate     = 1.0 if count > 100 else 0.0
        srv_serror_rate = serror_rate

        same_srv_rate   = (srv_count / count) if count else 1.0
        diff_srv_rate   = 1.0 - same_srv_rate

        method_map  = {"GET": 0, "POST": 1, "PUT": 2, "DELETE": 3, "PATCH": 4}
        method_code = method_map.get(method, 5)

        # Parse IP octets for dst_host features
        try:
            parts  = [int(x) for x in ip.split(".")]
            octets = parts + [0] * (4 - len(parts))
        except Exception:
            octets = [0, 0, 0, 0]

        # High request count = scan-like behaviour
        dst_host_count     = min(count * 2, 255)
        dst_host_srv_count = min(srv_count * 2, 255)

        dst_host_serror_rate     = serror_rate
        dst_host_srv_serror_rate = serror_rate
        dst_host_same_srv_rate   = same_srv_rate
        dst_host_diff_srv_rate   = diff_srv_rate

        return {
            "count":          count,
            "srv_count":      srv_count,
            "serror_rate":    round(serror_rate, 3),
            "same_srv_rate":  round(same_srv_rate, 3),
            "diff_srv_rate":  round(diff_srv_rate, 3),
            "src_bytes":      src_bytes,
        }

    def build_feature_vector(self, ip: str, path: str, method: str,
                             content_len: int, n_headers: int,
                             has_auth: int, has_ua: int,
                             qs_len: int) -> np.ndarray:
        """Assemble the full 41-feature KDD vector from live data."""
        now = time.time()
        cutoff = now - self.WINDOW

        with self._lock:
            dq = self._log[ip]
            window = [(ts, p, m, cl) for ts, p, m, cl, _ in dq if ts >= cutoff]

        count     = max(len(window), 1)
        srv_count = max(sum(1 for _, p, _, _ in window if p == path), 1)

        serror_rate  = 1.0 if count > 80 else (0.5 if count > 40 else 0.0)
        same_srv     = srv_count / count
        diff_srv     = 1.0 - same_srv

        method_map  = {"GET": 0, "POST": 1, "PUT": 2, "DELETE": 3, "PATCH": 4}
        method_code = method_map.get(method, 5)

        try:
            parts  = [int(x) for x in ip.split(".")]
            octets = parts + [0] * (4 - len(parts))
        except Exception:
            octets = [127, 0, 0, 1]

        dst_count = min(count * 2, 255)
        dst_srv   = min(srv_count * 2, 255)
        path_depth = path.count("/")

        features = [
            0,                    # duration
            method_code,          # protocol_type  (HTTP method as proxy)
            path_depth,           # service        (path depth as proxy)
            has_auth,             # flag
            content_len,          # src_bytes
            0,                    # dst_bytes
            0,                    # land
            0,                    # wrong_fragment
            0,                    # urgent
            n_headers,            # hot
            0,                    # num_failed_logins
            has_ua,               # logged_in
            0,                    # num_compromised
            0,                    # root_shell
            0,                    # su_attempted
            0,                    # num_root
            0,                    # num_file_creations
            0,                    # num_shells
            0,                    # num_access_files
            0,                    # num_outbound_cmds
            0,                    # is_host_login
            0,                    # is_guest_login
            count,                # count          ← REAL: requests in 2s
            srv_count,            # srv_count      ← REAL: same-path requests
            serror_rate,          # serror_rate    ← REAL: flood proxy
            serror_rate,          # srv_serror_rate
            0.0,                  # rerror_rate
            0.0,                  # srv_rerror_rate
            same_srv,             # same_srv_rate  ← REAL
            diff_srv,             # diff_srv_rate  ← REAL
            0.0,                  # srv_diff_host_rate
            dst_count,            # dst_host_count ← REAL
            dst_srv,              # dst_host_srv_count ← REAL
            same_srv,             # dst_host_same_srv_rate
            diff_srv,             # dst_host_diff_srv_rate
            0.0,                  # dst_host_same_src_port_rate
            0.0,                  # dst_host_srv_diff_host_rate
            serror_rate,          # dst_host_serror_rate ← REAL
            serror_rate,          # dst_host_srv_serror_rate ← REAL
            0.0,                  # dst_host_rerror_rate
            0.0,                  # dst_host_srv_rerror_rate
        ]

        return np.array(features, dtype=float).reshape(1, -1)


tracker = ConnectionTracker()


# ─────────────────────────────────────────────
#  Layer 1: IP Blacklist (rule-based)
# ─────────────────────────────────────────────

BLACKLISTED_IPS = set()   # تضاف تلقائياً لما الـ ML يكتشف هجوم

class IPBlocker:
    def __init__(self):
        self.blocked_until: dict[str, float] = {}

    def is_blocked(self, ip: str) -> bool:
        expires_at = self.blocked_until.get(ip)
        if expires_at is None:
            return False
        if time.time() >= expires_at:
            self.blocked_until.pop(ip, None)
            return False
        return True

    def block(self, ip: str, duration_seconds: int = 30):
        self.blocked_until[ip] = time.time() + duration_seconds
        print(f"[BLACKLIST] {ip} blocked for {duration_seconds}s")

    def unblock(self, ip: str):
        self.blocked_until.pop(ip, None)

ip_blocker = IPBlocker()

# ─────────────────────────────────────────────
#  Detector
# ─────────────────────────────────────────────

class Detector:

    def __init__(self):
        print("Loading model ...")
        self.model = xgb.XGBClassifier()
        self.model.load_model("models/xgboost_model.json")
        with open("models/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)
        print("Detector ready")

    def predict_request(self, req) -> dict:
        ip          = req.remote_addr or "127.0.0.1"
        path        = req.path
        method      = req.method
        content_len = req.content_length or 0
        n_headers   = len(list(req.headers))
        has_auth    = int("Authorization" in req.headers)
        has_ua      = int("User-Agent" in req.headers)
        qs_len      = len(req.query_string)

        # Record this connection BEFORE building features
        tracker.record(ip, path, method, content_len)

        # Build feature vector from REAL live traffic data
        X        = tracker.build_feature_vector(
                       ip, path, method, content_len,
                       n_headers, has_auth, has_ua, qs_len)
        # Keep feature names to avoid sklearn warning:
        # "X does not have valid feature names..."
        X_df = pd.DataFrame(X, columns=FEATURE_NAMES)
        X_scaled = self.scaler.transform(X_df)

        t0    = time.perf_counter()
        pred  = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]
        ms    = round((time.perf_counter() - t0) * 1000, 2)

        is_attack  = bool(pred)
        confidence = float(proba[int(pred)])

        # Pull the real features that drove the decision
        kdd = tracker.get_kdd_features(ip, path, method, content_len)
        attack_type = self.estimate_attack_type(kdd) if is_attack else "normal"

        return {
            "is_attack":    is_attack,
            "label":        "attack" if is_attack else "normal",
            "attack_type":  attack_type,
            "confidence":   round(confidence * 100, 1),
            "attack_prob":  round(float(proba[1]) * 100, 1),
            "normal_prob":  round(float(proba[0]) * 100, 1),
            "inference_ms": ms,
            "source_ip":    ip,
            "live_features": kdd,   # expose real computed features to frontend
        }

    @staticmethod
    def estimate_attack_type(kdd: dict) -> str:
        """
        Estimate attack family using NSL-KDD-style categories.
        This keeps binary model output, but adds human-readable type.
        """
        count = kdd.get("count", 0)
        serror_rate = kdd.get("serror_rate", 0.0)
        same_srv_rate = kdd.get("same_srv_rate", 1.0)
        diff_srv_rate = kdd.get("diff_srv_rate", 0.0)

        # DoS / DDoS pattern: very high request volume + high error rate
        if count >= 80 and serror_rate >= 0.5:
            return "DoS/DDoS"

        # Probe pattern: many requests to different services/paths
        if count >= 30 and diff_srv_rate >= 0.6:
            return "Probe"

        # R2L pattern proxy: low-volume but suspicious behavior
        if count < 20 and same_srv_rate < 0.5:
            return "R2L"

        # U2R pattern proxy: suspicious but not scan/flood-like
        return "U2R/Unknown"


detector = Detector()


# ─────────────────────────────────────────────
#  Middleware
# ─────────────────────────────────────────────

@app.before_request
def intrusion_middleware():
    g.request_id = str(uuid.uuid4())[:8]
    ip = request.remote_addr or "127.0.0.1"

    # Layer 1 — IP blacklist check (instant, no ML needed)
    if ip_blocker.is_blocked(ip):
        g.threat = {
            "is_attack": True, "label": "blocked",
            "attack_type": "blocked-ip",
            "confidence": 100.0, "attack_prob": 100.0,
            "normal_prob": 0.0, "inference_ms": 0.0,
            "source_ip": ip, "blocked_by": "blacklist",
            "live_features": {}
        }
        return  # middleware يوقف هون

    # Layer 2 — XGBoost ML detection
    g.threat = detector.predict_request(request)

    # إذا الـ ML اكتشف هجوم، يضيف الـ IP للـ blacklist تلقائياً
    # if g.threat["is_attack"]:
    #     ip_blocker.block(ip)

# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

@app.get("/")
def index():
    return jsonify({
        "status":     "online",
        "request_id": g.request_id,
        "threat":     g.threat,
    })

@app.route("/<path:any_path>", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
def catch_all(any_path):
    """
    Catch unknown paths so demo scenarios (e.g. /login, /scan/*)
    are inspected by middleware instead of returning noisy 404 logs.
    """
    if request.method == "OPTIONS":
        return ("", 204)

    if g.threat["is_attack"]:
        return jsonify({
            "error": "Blocked by NIDS",
            "path": "/" + any_path,
            "request_id": g.request_id,
            "threat": g.threat,
        }), 403

    return jsonify({
        "message": "Request observed",
        "path": "/" + any_path,
        "request_id": g.request_id,
        "threat": g.threat,
    }), 200


@app.get("/api/data")
def protected_data():
    if g.threat["is_attack"]:
        return jsonify({
            "error":      "Blocked by NIDS",
            "request_id": g.request_id,
            "threat":     g.threat,
        }), 403

    return jsonify({
        "message":    "Access granted",
        "request_id": g.request_id,
        "threat":     g.threat,
        "data":       {"records": 42, "server": "flask-nids"},
    })


@app.get("/api/status")
def api_status():
    return jsonify({
        "model":    "XGBoost · NSL-KDD · 41 features",
        "threat":   g.threat,
    })

@app.get("/api/admin/blacklist")
def get_blacklist():
    now = time.time()
    active_blocks = {
        ip: max(0, round(expires_at - now, 1))
        for ip, expires_at in ip_blocker.blocked_until.items()
        if expires_at > now
    }
    return jsonify({"blocked_ips": active_blocks})

@app.delete("/api/admin/blacklist/<ip>")
def unblock_ip(ip):
    ip_blocker.unblock(ip)
    return jsonify({"unblocked": ip})


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print("\n" + "=" * 60)
    print("  Network Intrusion Detection System")
    print(f"  Local:   http://127.0.0.1:5000")
    print(f"  Network: http://{local_ip}:5000   ← use this on other devices")
    print("=" * 60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)