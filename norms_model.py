# -*- coding: utf-8 -*-
"""
Параметрическая модель трудоёмкости сборки ШАУ.

Это Python-эквивалент калькулятора (Калькулятор_норм_ШАУ.xlsx). Используется:
  * как baseline и «разметчик» (Слой A);
  * как генератор «истинной» трудоёмкости для синтетического датасета (make_dataset.py);
  * как признак-подсказка для ML-модели (обучение на остатках).

Нормативы откалиброваны так, что эталонный (Anchor) шкаф = ~20 ч (требование ТЗ 18-22 ч).
"""

# Нормативы, ч/ед (совпадают с Excel-калькулятором)
NORMS = {
    "base": 1.0, "mount_panel": 0.4, "din_inclined": 0.3, "hole": 0.07,
    "breaker": 0.18, "load_switch": 0.25, "contactor": 0.22, "rcd": 0.20,
    "freq_mount": 0.45, "soft": 0.45, "psu": 0.25, "phase_relay": 0.18,
    "relay": 0.12, "fan": 0.22, "thermostat": 0.18, "fuse": 0.08,
    "plc": 0.6, "hmi": 0.8, "lamp": 0.18, "analog_mount": 0.22, "iface": 0.25,
    "terminal": 0.055, "marking": 0.02,
    "pnr_base": 0.5, "pnr_analog": 0.2,
}

# Числовые признаки, подаваемые в модель
NUM_FEATURES = [
    "width_mm", "height_mm", "depth_mm", "mount_panel", "din_inclined", "cable_entries",
    "breakers_le63", "load_switches", "contactors", "rcd", "freq_converters",
    "soft_starters", "power_supplies", "phase_relays", "relays", "fans", "thermostats",
    "fuse_holders", "signal_lamps", "analog_4_20mA", "interfaces_count", "terminals",
]
# Категориальные признаки
CAT_FEATURES = ["ip_rating", "plc_type", "hmi_type"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

IP_LEVELS = ["IP21", "IP54", "IP65", "IP66"]
PLC_TYPES = ["Нет", "Релейное", "Компактный ПЛК", "Модульный ПЛК"]
HMI_TYPES = ["Нет", "До 7\"", "Свыше 7\""]


def _g(f, k, default=0):
    v = f.get(k, default)
    return default if v is None else v


def estimate_hours(f, breakdown=False):
    """Параметрическая оценка трудоёмкости, ч. f — словарь признаков."""
    N = NORMS
    mech = (N["base"] + _g(f, "mount_panel") * N["mount_panel"]
            + _g(f, "din_inclined") * N["din_inclined"]
            + _g(f, "cable_entries") * N["hole"])
    power = (_g(f, "breakers_le63") * N["breaker"] + _g(f, "load_switches") * N["load_switch"]
             + _g(f, "contactors") * N["contactor"] + _g(f, "rcd") * N["rcd"]
             + _g(f, "freq_converters") * N["freq_mount"] + _g(f, "soft_starters") * N["soft"]
             + _g(f, "power_supplies") * N["psu"] + _g(f, "phase_relays") * N["phase_relay"]
             + _g(f, "relays") * N["relay"] + _g(f, "fans") * N["fan"]
             + _g(f, "thermostats") * N["thermostat"] + _g(f, "fuse_holders") * N["fuse"])
    plc_w = {"Нет": 0, "Релейное": 0.5, "Компактный ПЛК": 1.0, "Модульный ПЛК": 1.6}
    hmi_w = {"Нет": 0, "До 7\"": 1.0, "Свыше 7\"": 1.4}
    low = (plc_w.get(f.get("plc_type", "Нет"), _g(f, "plc")) * N["plc"]
           + hmi_w.get(f.get("hmi_type", "Нет"), _g(f, "hmi")) * N["hmi"]
           + _g(f, "signal_lamps") * N["lamp"]
           + _g(f, "analog_4_20mA") * N["analog_mount"]
           + _g(f, "interfaces_count") * N["iface"])
    term = _g(f, "terminals") * N["terminal"] + _g(f, "terminals") * N["marking"]
    n_pch = _g(f, "freq_converters")
    pnr_pch = 3.0 if n_pch >= 2 else n_pch * 0.5
    pnr = N["pnr_base"] + _g(f, "analog_4_20mA") * N["pnr_analog"] + pnr_pch
    ip_extra = mech * 0.30 if f.get("ip_rating") == "IP66" else 0.0
    total = mech + power + low + term + pnr + ip_extra
    if breakdown:
        return {"Мехобработка": mech, "Силовой монтаж": power, "Слаботочка": low,
                "Клеммы+маркировка": term, "ПНР": pnr, "IP66": ip_extra, "ИТОГО": total}
    return total


# Эталонный (Anchor) шкаф из ТЗ / приложенного примера
ANCHOR = {
    "ip_rating": "IP66", "width_mm": 800, "height_mm": 600, "depth_mm": 300,
    "mount_panel": 1, "din_inclined": 1, "cable_entries": 19,
    "breakers_le63": 5, "load_switches": 1, "contactors": 0, "rcd": 0,
    "freq_converters": 3, "soft_starters": 0, "power_supplies": 1, "phase_relays": 1,
    "relays": 3, "fans": 2, "thermostats": 1, "fuse_holders": 7, "signal_lamps": 1,
    "analog_4_20mA": 6, "interfaces_count": 2, "terminals": 50,
    "plc_type": "Компактный ПЛК", "hmi_type": "До 7\"",
}

if __name__ == "__main__":
    bd = estimate_hours(ANCHOR, breakdown=True)
    for k, v in bd.items():
        print(f"{k:20}: {v:5.2f}")
