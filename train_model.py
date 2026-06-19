# -*- coding: utf-8 -*-
"""
Прототип обучения ML-модели нормирования трудоёмкости ШАУ (Слой C).

Основной движок — CatBoost (рекомендован ТЗ для табличных задач с категориями).
Если CatBoost/sklearn недоступны (нет интернета для установки), автоматически
используется запасной градиентный бустинг на numpy (gbdt_numpy.py) — чтобы конвейер
запускался в любом окружении.

Что делает:
  1) грузит датасет (по умолчанию synthetic_dataset.csv от make_dataset.py);
  2) train/val split;
  3) обучение на остатках: целевая = (факт − baseline) — модель учит ПОПРАВКУ к
     параметрической оценке Слоя A (устойчивее на малых данных);
  4) weighted loss — повышенный вес редким сложным шкафам (IP66, каскад ПЧ);
  5) метрики MAE / MAPE / R², сравнение с baseline;
  6) тест на эталонном (Anchor) шкафу — требование ТЗ 18-22 ч;
  7) важности признаков; сохранение модели.

Запуск:  python3 train_model.py --data synthetic_dataset.csv
"""
import argparse
import json
import numpy as np
import pandas as pd

from norms_model import NUM_FEATURES, CAT_FEATURES, ALL_FEATURES, ANCHOR, estimate_hours

TARGET = "actual_hours"
BASELINE = "baseline_hours"


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    mae = float(np.mean(np.abs(y - p)))
    mape = float(np.mean(np.abs(y - p) / y) * 100)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return mae, mape, r2


def sample_weights(df):
    """Больше вес редким/сложным шкафам — аналог weighted loss."""
    w = np.ones(len(df))
    w[df["ip_rating"] == "IP66"] *= 2.0
    w[df["freq_converters"] >= 2] *= 1.5
    w[df["plc_type"] == "Модульный ПЛК"] *= 1.3
    return w


def encode_categories(df, maps=None):
    """Ординальное кодирование категорий (для numpy-движка)."""
    df = df.copy()
    if maps is None:
        maps = {c: {v: i for i, v in enumerate(sorted(df[c].astype(str).unique()))}
                for c in CAT_FEATURES}
    for c in CAT_FEATURES:
        df[c] = df[c].astype(str).map(lambda v: maps[c].get(v, -1))
    return df, maps


def train_catboost(Xtr, ytr, wtr, Xval, cat_idx):
    from catboost import CatBoostRegressor, Pool
    model = CatBoostRegressor(
        iterations=600, learning_rate=0.05, depth=6, loss_function="RMSE",
        l2_leaf_reg=3.0, random_seed=42, verbose=False)
    pool_tr = Pool(Xtr, ytr, cat_features=cat_idx, weight=wtr)
    model.fit(pool_tr)
    return model, model.predict(Pool(Xval, cat_features=cat_idx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic_dataset.csv")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out", default="model_shau.json")
    args = ap.parse_args()

    df = pd.read_csv(args.data)

    # --- Очистка выбросов (ТЗ, Этап 1): отсев записей, где факт сильно расходится с
    #     оценкой не из-за сложности, а из-за внешних причин (простои поставок и т.п.). ---
    n0 = len(df)
    ratio = df[TARGET] / df[BASELINE]
    lo, hi = ratio.quantile(0.02), ratio.quantile(0.98)
    df = df[(ratio >= lo) & (ratio <= hi)].reset_index(drop=True)
    print(f"Очистка выбросов: удалено {n0 - len(df)} из {n0} записей "
          f"(факт/оценка вне [{lo:.2f}; {hi:.2f}]).\n")

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    n_val = int(len(df) * args.val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    tr, val = df.iloc[tr_idx].copy(), df.iloc[val_idx].copy()

    # Обучение на остатках: цель = факт − baseline
    ytr = (tr[TARGET] - tr[BASELINE]).values
    wtr = sample_weights(tr)

    engine = None
    try:
        cat_idx = [ALL_FEATURES.index(c) for c in CAT_FEATURES]
        model, pred_resid = train_catboost(
            tr[ALL_FEATURES], ytr, wtr, val[ALL_FEATURES], cat_idx)
        engine = "CatBoost"
        importances = dict(zip(ALL_FEATURES, model.get_feature_importance()))
        anchor_feat = pd.DataFrame([{**ANCHOR}])[ALL_FEATURES]
        anchor_resid = float(model.predict(anchor_feat)[0])
    except Exception as e:
        # ---- ЗАПАСНОЙ движок (numpy) ----
        from gbdt_numpy import GBDTRegressor
        tr_e, maps = encode_categories(tr)
        val_e, _ = encode_categories(val, maps)
        model = GBDTRegressor(n_estimators=450, learning_rate=0.04, max_depth=4,
                              min_samples=12, subsample=0.8, random_state=42)
        model.fit(tr_e[ALL_FEATURES].values, ytr, sample_weight=wtr)
        pred_resid = model.predict(val_e[ALL_FEATURES].values)
        engine = "numpy-GBDT (fallback)"
        importances = dict(zip(ALL_FEATURES, model.feature_importances_))
        anchor_e, _ = encode_categories(pd.DataFrame([{**ANCHOR}]), maps)
        anchor_resid = float(model.predict(anchor_e[ALL_FEATURES].values)[0])
        print(f"[i] CatBoost недоступен ({type(e).__name__}); использован запасной движок.\n")

    # Прогноз = baseline + поправка
    pred_ml = val[BASELINE].values + pred_resid
    y_val = val[TARGET].values

    mae_b, mape_b, r2_b = metrics(y_val, val[BASELINE].values)   # только Слой A
    mae_m, mape_m, r2_m = metrics(y_val, pred_ml)                # Слой A + ML

    print(f"=== Движок: {engine} ===")
    print(f"Датасет: {len(df)} проектов (train {len(tr)} / val {len(val)})\n")
    print(f"{'Модель':28} {'MAE, ч':>8} {'MAPE, %':>9} {'R²':>7}")
    print(f"{'Параметрическая (Слой A)':28} {mae_b:8.2f} {mape_b:9.1f} {r2_b:7.3f}")
    print(f"{'Слой A + ML-поправка':28} {mae_m:8.2f} {mape_m:9.1f} {r2_m:7.3f}")

    # Тест на Anchor
    anchor_base = estimate_hours(ANCHOR)
    anchor_pred = anchor_base + anchor_resid
    ok = 18 <= anchor_pred <= 22
    print(f"\nТест на Anchor-проекте (ТЗ: 18-22 ч):")
    print(f"  baseline={anchor_base:.1f} ч  +поправка={anchor_resid:+.1f} ч  "
          f"=> прогноз={anchor_pred:.1f} ч  [{'OK' if ok else 'ВНЕ ДИАПАЗОНА'}]")

    print(f"\nКритерии приёмки ТЗ: MAE ≤10-15%, R² ≥0.85")
    crit = (mape_m <= 15) and (r2_m >= 0.85)
    print(f"  MAPE={mape_m:.1f}%  R²={r2_m:.3f}  =>  {'СОБЛЮДЕНЫ' if crit else 'не соблюдены'}")

    print("\nТоп-10 важных признаков:")
    for k, v in sorted(importances.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k:22} {v:6.3f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"engine": engine,
                   "metrics_ml": {"MAE": mae_m, "MAPE": mape_m, "R2": r2_m},
                   "metrics_baseline": {"MAE": mae_b, "MAPE": mape_b, "R2": r2_b},
                   "anchor_pred_h": anchor_pred,
                   "importances": importances}, f, ensure_ascii=False, indent=2)
    print(f"\nОтчёт сохранён: {args.out}")


if __name__ == "__main__":
    main()
