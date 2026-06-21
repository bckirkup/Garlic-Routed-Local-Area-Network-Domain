"""Biometric profile definitions and population generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BiometricProfile:
    """Resting biometric profile for a single agent.

    Values drawn from physiologically plausible distributions matching
    OpenWearables schema conventions.
    """

    resting_hr: float  # beats/min
    resting_hrv: float  # RMSSD ms
    resting_rr: float  # breaths/min
    resting_temp: float  # °C core body temp
    hr_circadian_amp: float = 0.0
    rr_circadian_amp: float = 0.0
    temp_circadian_amp: float = 0.0


def generate_profiles(n: int, rng: np.random.Generator) -> list[BiometricProfile]:
    """Generate physiologically plausible biometric profiles for n agents."""
    hrs = rng.normal(72.0, 8.0, n).clip(50, 110)
    hrvs = rng.normal(42.0, 15.0, n).clip(10, 120)
    rrs = rng.normal(15.0, 2.0, n).clip(8, 25)
    temps = rng.normal(36.8, 0.3, n).clip(35.5, 38.0)
    hr_ca = rng.uniform(3.0, 8.0, n)
    rr_ca = rng.uniform(0.5, 2.0, n)
    temp_ca = rng.uniform(0.2, 0.5, n)

    return [
        BiometricProfile(
            resting_hr=float(hrs[i]),
            resting_hrv=float(hrvs[i]),
            resting_rr=float(rrs[i]),
            resting_temp=float(temps[i]),
            hr_circadian_amp=float(hr_ca[i]),
            rr_circadian_amp=float(rr_ca[i]),
            temp_circadian_amp=float(temp_ca[i]),
        )
        for i in range(n)
    ]


def circadian_factor(hour_of_day: float) -> float:
    """Circadian modulation factor [-1, 1] peaking around 14:00 (2 PM)."""
    return float(np.sin(2 * np.pi * (hour_of_day - 2.0) / 24.0))


def seasonal_factor(day_of_year: int) -> float:
    """Seasonal baseline shift factor [-1, 1], peak in summer (day 172)."""
    return float(np.sin(2 * np.pi * (day_of_year - 80) / 365.0))
