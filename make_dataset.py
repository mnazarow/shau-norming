# -*- coding: utf-8 -*-
"""
Генератор синтетического датасета проектов ШАУ.

Зачем: на старте у предприятия ещё нет размеченного архива. Чтобы собрать и проверить
ML-конвейер (train_model.py) уже сейчас, генерируем правдоподобные конфигурации шкафов
и считаем «фактическую» трудоёмкость как:

    факт = параметрическая_оценка(признаки)            # Слой A
           * лог-нормальный шум (~8%)                   # разброс бригад/условий
           + структурные нелинейности                  # чего baseline НЕ знает
           ( + редкие выбросы )

Структурные нелинейности (их и должна «выучить» ML-модель сверх baseline):
  * IP66 + много вводов        -> доп. время на герметизацию (взаимодействие);
  * большие клеммники          -> эффект обучения (сублинейность);
  * каскад ПЧ + модульный ПЛК  -> доп. сложность ПНР.

ВНИМАНИЕ: данные синтетические — для отладки конвейера. На проде заменяются выгрузкой
факт-часов из 1С.

Запуск:  python3 make_dataset.py --n 800 --out synthetic_dataset.csv
"""
import argparse
import numpy as np
import pandas as pd

from norms_model import (estimate_hours, ALL_FEATURES, IP_LEVELS, PLC_TYPES, HMI_TYPES)

DIMS = [(400, 300, 200), (600, 400, 250), (600, 600, 250), (800, 600, 300),
        (800, 800, 300), (1000, 800, 300), (1200, 1000, 400), (2000, 800, 600)]


def sample_cabinet(rng):
    """Сэмплирует одну правдоподобную конфигурацию шкафа (признаки коррелированы)."""
    # масштаб проекта задаётся числом приводов/двигателей
    motors = rng.integers(0, 9)                       # 0..8 двигателей
    has_vfd = rng.random() < 0.6 and motors > 0
    n_vfd = int(min(motors, rng.integers(1, motors + 1))) if has_vfd else 0
    n_soft = int(rng.integers(0, max(1, motors - n_vfd + 1))) if (motors and rng.random() < 0.2) else 0

    ip = rng.choice(IP_LEVELS, p=[0.20, 0.40, 0.25, 0.15])   # IP66 — реже
    w, h, d = DIMS[rng.integers(0, len(DIMS))]

    breakers = int(motors + rng.integers(2, 6))              # вводные + отходящие
    contactors = int(max(0, motors - n_vfd) + rng.integers(0, 3))
    rcd = int(rng.integers(0, 3))
    relays = int(rng.integers(0, motors + 4))
    psu = int(rng.random() < 0.85)
    phase_relay = int((w >= 600) and rng.random() < 0.7)
    fans = int(rng.random() < 0.5) * rng.integers(1, 3)
    thermostats = int(fans > 0)
    fuse_holders = int(rng.integers(0, 8))
    lamps = int(rng.integers(0, 6))

    # автоматика
    plc_type = rng.choice(PLC_TYPES, p=[0.25, 0.20, 0.40, 0.15])
    hmi_type = rng.choice(HMI_TYPES, p=[0.45, 0.40, 0.15])
    analog = int(rng.integers(0, 12)) if plc_type != "Нет" else 0
    ifaces = 0
    if plc_type != "Нет":
        ifaces = int(rng.integers(1, 4))

    # коммутационный объём — растёт с числом аппаратов и аналоговых цепей
    base_terms = breakers * 4 + contactors * 3 + analog * 2 + relays * 2 + 6
    terminals = int(base_terms * rng.uniform(0.8, 1.3))
    cable_entries = int(rng.integers(2, 6) + motors + (analog // 2))

    mount_panel = int(rng.random() < 0.95)
    din_inclined = int(rng.random() < 0.3)

    return {
        "ip_rating": ip, "width_mm": w, "height_mm": h, "depth_mm": d,
        "mount_panel": mount_panel, "din_inclined": din_inclined,
        "cable_entries": cable_entries, "breakers_le63": breakers,
        "load_switches": int(rng.integers(0, 2)), "contactors": contactors, "rcd": rcd,
        "freq_converters": n_vfd, "soft_starters": n_soft, "power_supplies": psu,
        "phase_relays": phase_relay, "relays": relays, "fans": int(fans),
        "thermostats": thermostats, "fuse_holders": fuse_holders, "signal_lamps": lamps,
        "analog_4_20mA": analog, "interfaces_count": ifaces, "terminals": terminals,
        "plc_type": plc_type, "hmi_type": hmi_type,
    }


def true_hours(f, rng):
    """«Фактическая» трудоёмкость = baseline + структурные нелинейности + шум."""
    base = estimate_hours(f)
    extra = 0.0
    # 1) IP66 + много вводов -> герметизация (взаимодействие), baseline этого не знает
    if f["ip_rating"] == "IP66":
        extra += 0.06 * f["cable_entries"]
    # 2) большие клеммники -> сублинейность (эффект обучения бригады)
    if f["terminals"] > 60:
        extra -= 0.02 * (f["terminals"] - 60)
    # 3) каскад ПЧ + модульный ПЛК -> сложная ПНР каскада
    if f["freq_converters"] >= 2 and f["plc_type"] == "Модульный ПЛК":
        extra += 2.0
    # 4) много аналоговых цепей -> нелинейный рост наладки
    if f["analog_4_20mA"] > 6:
        extra += 0.30 * (f["analog_4_20mA"] - 6)
    # 5) глубокий шкаф + плотная компоновка -> доп. время монтажа
    if f["depth_mm"] >= 400 and f["terminals"] > 50:
        extra += 1.0
    h = (base + extra) * rng.lognormal(mean=0.0, sigma=0.05)
    # редкие выбросы (как в реальных данных 1С)
    if rng.random() < 0.02:
        h *= rng.uniform(1.3, 1.7)
    return max(2.0, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--out", default="synthetic_dataset.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for _ in range(args.n):
        f = sample_cabinet(rng)
        f["actual_hours"] = round(true_hours(f, rng), 2)
        f["baseline_hours"] = round(estimate_hours(f), 2)   # оценка Слоя A (как признак)
        rows.append(f)

    df = pd.DataFrame(rows)[ALL_FEATURES + ["baseline_hours", "actual_hours"]]
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"Сохранено: {args.out}  ({len(df)} строк, {df.shape[1]} столбцов)")
    print(f"Трудоёмкость, ч:  мин={df.actual_hours.min():.1f}  "
          f"медиана={df.actual_hours.median():.1f}  макс={df.actual_hours.max():.1f}")
    mape_base = (abs(df.actual_hours - df.baseline_hours) / df.actual_hours).mean() * 100
    print(f"MAPE baseline (Слой A) на синтетике: {mape_base:.1f}%  "
          f"(ML должна улучшить за счёт структурных эффектов)")


if __name__ == "__main__":
    main()
