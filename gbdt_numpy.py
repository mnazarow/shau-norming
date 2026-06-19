# -*- coding: utf-8 -*-
"""
Минимальный градиентный бустинг на регрессионных деревьях (только numpy).

Назначение — ЗАПАСНОЙ движок для train_model.py в окружении без CatBoost/sklearn
(нужен только для отладки конвейера). На проде используйте CatBoost — он точнее,
быстрее и нативно обрабатывает категориальные признаки.

Поддерживает веса наблюдений (sample_weight) — аналог weighted loss для редких
сложных шкафов (IP66, взрывозащита).
"""
import numpy as np


class _Tree:
    def __init__(self, max_depth=4, min_samples=8):
        self.max_depth = max_depth
        self.min_samples = min_samples

    def fit(self, X, y, w):
        self.node = self._build(X, y, w, 0)
        return self

    def _leaf(self, y, w):
        return float(np.sum(w * y) / max(np.sum(w), 1e-12))

    def _build(self, X, y, w, depth):
        if depth >= self.max_depth or len(y) < 2 * self.min_samples:
            return {"leaf": self._leaf(y, w)}
        feat, thr, gain = self._best_split(X, y, w)
        if feat is None or gain <= 1e-9:
            return {"leaf": self._leaf(y, w)}
        mask = X[:, feat] <= thr
        if mask.sum() < self.min_samples or (~mask).sum() < self.min_samples:
            return {"leaf": self._leaf(y, w)}
        return {"feat": feat, "thr": thr,
                "L": self._build(X[mask], y[mask], w[mask], depth + 1),
                "R": self._build(X[~mask], y[~mask], w[~mask], depth + 1)}

    def _best_split(self, X, y, w):
        n, d = X.shape
        Wy = w * y
        tot_wy, tot_w = Wy.sum(), w.sum()
        parent = tot_wy * tot_wy / max(tot_w, 1e-12)   # weighted SS of mean
        best = (None, None, 0.0)
        for f in range(d):
            xf = X[:, f]
            order = np.argsort(xf, kind="mergesort")
            xs, wy_s, w_s = xf[order], Wy[order], w[order]
            cwy, cw = np.cumsum(wy_s), np.cumsum(w_s)
            # допустимые точки разреза — где значение меняется
            diff = np.where(np.diff(xs) > 0)[0]
            if diff.size == 0:
                continue
            lw, lwy = cw[diff], cwy[diff]
            rw, rwy = tot_w - lw, tot_wy - lwy
            ok = (lw >= self.min_samples) & (rw >= self.min_samples)
            if not ok.any():
                continue
            score = lwy * lwy / np.maximum(lw, 1e-12) + rwy * rwy / np.maximum(rw, 1e-12)
            gain = score - parent
            gain[~ok] = -np.inf
            j = int(np.argmax(gain))
            if gain[j] > best[2]:
                thr = (xs[diff[j]] + xs[diff[j] + 1]) / 2.0
                best = (f, thr, float(gain[j]))
        return best

    def predict(self, X):
        return np.array([self._pred_row(r, self.node) for r in X])

    def _pred_row(self, r, node):
        while "leaf" not in node:
            node = node["L"] if r[node["feat"]] <= node["thr"] else node["R"]
        return node["leaf"]


class GBDTRegressor:
    def __init__(self, n_estimators=300, learning_rate=0.05, max_depth=4,
                 min_samples=8, subsample=0.8, random_state=0):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.subsample = subsample
        self.rng = np.random.default_rng(random_state)
        self.trees = []

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, float); y = np.asarray(y, float)
        n = len(y)
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
        self.base = float(np.sum(w * y) / np.sum(w))
        pred = np.full(n, self.base)
        for _ in range(self.n_estimators):
            resid = y - pred
            idx = self.rng.choice(n, int(self.subsample * n), replace=False)
            t = _Tree(self.max_depth, self.min_samples).fit(X[idx], resid[idx], w[idx])
            pred += self.lr * t.predict(X)
            self.trees.append(t)
        # суммарная важность признаков (по приросту, грубо — частоте использования)
        self.feature_importances_ = self._importance(X.shape[1])
        return self

    def _importance(self, d):
        imp = np.zeros(d)
        def walk(node):
            if "leaf" in node: return
            imp[node["feat"]] += 1
            walk(node["L"]); walk(node["R"])
        for t in self.trees:
            walk(t.node)
        s = imp.sum()
        return imp / s if s else imp

    def predict(self, X):
        X = np.asarray(X, float)
        pred = np.full(len(X), self.base)
        for t in self.trees:
            pred += self.lr * t.predict(X)
        return pred
