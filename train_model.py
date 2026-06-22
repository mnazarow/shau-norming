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
import os
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


def run_training(data_path="synthetic_dataset.csv", val_frac=0.2, out="model_shau.json"):
    """Обучает модель и возвращает (отчёт, предиктор-поправки).

    Отчёт — JSON-сериализуемый словарь с метриками, Anchor-тестом, важностями и
    предсказаниями на валидации (для графиков). Предиктор — функция features->поправка,
    которой пользуется веб-интерфейс для прогноза.
    """
    df = pd.read_csv(data_path)

    # Очистка выбросов (ТЗ, Этап 1): отсев записей, где факт расходится с оценкой не
    # из-за сложности, а из-за внешних причин (простои поставок и т.п.).
    n0 = len(df)
    ratio = df[TARGET] / df[BASELINE]
    lo, hi = ratio.quantile(0.02), ratio.quantile(0.98)
    df = df[(ratio >= lo) & (ratio <= hi)].reset_index(drop=True)
    n_out = n0 - len(df)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    n_val = int(len(df) * val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    tr, val = df.iloc[tr_idx].copy(), df.iloc[val_idx].copy()

    ytr = (tr[TARGET] - tr[BASELINE]).values
    wtr = sample_weights(tr)

    try:
        cat_idx = [ALL_FEATURES.index(c) for c in CAT_FEATURES]
        model, pred_resid = train_catboost(
            tr[ALL_FEATURES], ytr, wtr, val[ALL_FEATURES], cat_idx)
        engine = "CatBoost"
        importances = dict(zip(ALL_FEATURES, [float(x) for x in model.get_feature_importance()]))
        try:
            model.save_model("model_shau.cbm")
        except Exception:
            pass

        def predictor(features):
            row = pd.DataFrame([{k: features.get(k) for k in ALL_FEATURES}])
            return float(model.predict(row[ALL_FEATURES])[0])
    except Exception:
        from gbdt_numpy import GBDTRegressor
        tr_e, maps = encode_categories(tr)
        val_e, _ = encode_categories(val, maps)
        model = GBDTRegressor(n_estimators=450, learning_rate=0.04, max_depth=4,
                              min_samples=12, subsample=0.8, random_state=42)
        model.fit(tr_e[ALL_FEATURES].values, ytr, sample_weight=wtr)
        pred_resid = model.predict(val_e[ALL_FEATURES].values)
        engine = "numpy-GBDT (fallback)"
        importances = dict(zip(ALL_FEATURES, [float(x) for x in model.feature_importances_]))

        def predictor(features):
            row, _ = encode_categories(pd.DataFrame([{k: features.get(k) for k in ALL_FEATURES}]), maps)
            return float(model.predict(row[ALL_FEATURES].values)[0])

    pred_ml = val[BASELINE].values + pred_resid
    y_val = val[TARGET].values
    mae_b, mape_b, r2_b = metrics(y_val, val[BASELINE].values)
    mae_m, mape_m, r2_m = metrics(y_val, pred_ml)

    anchor_base = estimate_hours(ANCHOR)
    anchor_resid = predictor(ANCHOR)
    anchor_pred = anchor_base + anchor_resid

    report = {
        "engine": engine, "dataset": os.path.basename(data_path),
        "n_total": n0, "n_used": len(df), "n_outliers": n_out,
        "n_train": len(tr), "n_val": len(val),
        "metrics_baseline": {"MAE": mae_b, "MAPE": mape_b, "R2": r2_b},
        "metrics_ml": {"MAE": mae_m, "MAPE": mape_m, "R2": r2_m},
        "anchor": {"baseline": round(anchor_base, 2), "delta": round(anchor_resid, 2),
                   "pred": round(anchor_pred, 2), "ok": bool(18 <= anchor_pred <= 22)},
        "criteria_ok": bool(mape_m <= 15 and r2_m >= 0.85),
        "importances": importances,
        "val": {"actual": [round(v, 2) for v in y_val.tolist()],
                "baseline": [round(v, 2) for v in val[BASELINE].tolist()],
                "ml": [round(v, 2) for v in pred_ml.tolist()]},
    }
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    return report, predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="synthetic_dataset.csv")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--out", default="model_shau.json")
    args = ap.parse_args()

    r, _ = run_training(args.data, args.val_frac, args.out)
    print(f"=== Движок: {r['engine']} ===")
    print(f"Очистка выбросов: удалено {r['n_outliers']} из {r['n_total']}")
    print(f"Датасет: {r['n_used']} проектов (train {r['n_train']} / val {r['n_val']})\n")
    print(f"{'Модель':28} {'MAE, ч':>8} {'MAPE, %':>9} {'R²':>7}")
    mb, mm = r["metrics_baseline"], r["metrics_ml"]
    print(f"{'Параметрическая (Слой A)':28} {mb['MAE']:8.2f} {mb['MAPE']:9.1f} {mb['R2']:7.3f}")
    print(f"{'Слой A + ML-поправка':28} {mm['MAE']:8.2f} {mm['MAPE']:9.1f} {mm['R2']:7.3f}")
    a = r["anchor"]
    print(f"\nAnchor-тест (18-22 ч): baseline {a['baseline']} + поправка {a['delta']:+} "
          f"=> {a['pred']} ч  [{'OK' if a['ok'] else 'ВНЕ ДИАПАЗОНА'}]")
    print(f"Критерии ТЗ (MAPE≤15%, R²≥0.85): {'СОБЛЮДЕНЫ' if r['criteria_ok'] else 'не соблюдены'}")
    print("\nТоп-10 важных признаков:")
    for k, v in sorted(r["importances"].items(), key=lambda x: -x[1])[:10]:
        print(f"  {k:22} {v:6.3f}")
    print(f"\nОтчёт сохранён: {args.out}")


if __name__ == "__main__":
    main()
