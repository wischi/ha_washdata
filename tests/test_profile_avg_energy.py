# WashData - Home Assistant integration for appliance cycle monitoring via smart plugs.
# Copyright (C) 2026 Lukas Bandura
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for profile average energy (avg_energy) calculation.

Ensures:
1. Real cycles' average energy is calculated directly from cycle report energies
   rather than integrating the DTW-averaged power curve (which inflates energy
   due to peak smearing).
2. Hardware cumulative meter readings (energy_meter_wh, issue #316) take priority
   over trapezoidal power integration (energy_wh).
3. Reference-only profiles do not report phantom cycle_count.
"""
import asyncio
import inspect
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from custom_components.ha_washdata.profile_store import ProfileStore

BASE = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_hass(tmp_path):
    hass = MagicMock()
    hass.config.path = lambda *a: str(tmp_path.joinpath(*a))

    async def _exec(func, *args, **kwargs):
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return func(*args, **kwargs)

    hass.async_add_executor_job = AsyncMock(side_effect=_exec)
    hass.async_create_task = lambda coro, *a: asyncio.create_task(coro)
    return hass


@pytest.fixture
def store(mock_hass):
    with patch("custom_components.ha_washdata.profile_store.WashDataStore"):
        ps = ProfileStore(mock_hass, "test_entry", min_duration_ratio=0.0, max_duration_ratio=3.0)
        ps._store.async_load = AsyncMock(return_value=None)
        ps._store.async_save = AsyncMock()
        yield ps


def _flat_trace(watts: float, dur: int = 1800, n: int = 31):
    step = dur / (n - 1)
    return [((BASE + timedelta(seconds=i * step)).isoformat(), float(watts)) for i in range(n)]


@pytest.mark.asyncio
async def test_avg_energy_uses_real_cycles_without_reference_cycles(store):
    """Real cycles without reference cycles must calculate avg_energy from cycle energies."""
    # 2 cycles at 300 W for 3600 s -> 300 Wh = 0.3 kWh each
    for i in range(2):
        await store.async_add_cycle({
            "start_time": (BASE + timedelta(hours=i * 2)).isoformat(),
            "duration": 3600,
            "status": "completed",
            "profile_name": "Bob Eco",
            "power_data": _flat_trace(300, dur=3600, n=61),
            "energy_wh": 300.0,
        })

    await store.async_rebuild_envelope("Bob Eco")
    env = store.get_envelope("Bob Eco")

    assert env is not None
    assert env["cycle_count"] == 2
    assert env["avg_energy"] == pytest.approx(0.3, rel=1e-3)


@pytest.mark.asyncio
async def test_avg_energy_prioritizes_energy_meter_wh(store):
    """External meter delta (energy_meter_wh, issue #316) has priority over integrated energy_wh."""
    # Machine integrated energy_wh is ~500 Wh, but smart plug meter recorded 300 Wh and 310 Wh
    await store.async_add_cycle({
        "start_time": BASE.isoformat(),
        "duration": 3600,
        "status": "completed",
        "profile_name": "Bob Eco",
        "power_data": _flat_trace(500, dur=3600, n=61),
        "energy_wh": 500.0,
        "energy_meter_wh": 300.0,
        "energy_source": "meter",
    })
    await store.async_add_cycle({
        "start_time": (BASE + timedelta(hours=2)).isoformat(),
        "duration": 3600,
        "status": "completed",
        "profile_name": "Bob Eco",
        "power_data": _flat_trace(520, dur=3600, n=61),
        "energy_wh": 520.0,
        "energy_meter_wh": 310.0,
        "energy_source": "meter",
    })

    await store.async_rebuild_envelope("Bob Eco")
    env = store.get_envelope("Bob Eco")

    assert env is not None
    assert env["cycle_count"] == 2
    # Expect (0.300 + 0.310) / 2 = 0.305 kWh, NOT (0.500 + 0.520) / 2 = 0.510 kWh
    assert env["avg_energy"] == pytest.approx(0.305, rel=1e-3)


@pytest.mark.asyncio
async def test_avg_energy_not_distorted_by_dtw_shifted_pulses(store):
    """Pulsed heating at different offsets in different runs must not inflate avg_energy via DTW smearing."""
    # Cycle 1: 1800s, 2000W pulse for 300s at start (energy = 2000 * 300 / 3600 = 166.67 Wh)
    trace1 = []
    for t in range(0, 1801, 30):
        w = 2000.0 if 120 <= t <= 420 else 5.0
        trace1.append(((BASE + timedelta(seconds=t)).isoformat(), w))

    # Cycle 2: 1800s, 2000W pulse for 300s near middle (energy = 2000 * 300 / 3600 = 166.67 Wh)
    trace2 = []
    for t in range(0, 1801, 30):
        w = 2000.0 if 900 <= t <= 1200 else 5.0
        trace2.append(((BASE + timedelta(hours=2, seconds=t)).isoformat(), w))

    await store.async_add_cycle({
        "start_time": BASE.isoformat(),
        "duration": 1800,
        "status": "completed",
        "profile_name": "Bob Express",
        "power_data": trace1,
        "energy_wh": 170.0,
    })
    await store.async_add_cycle({
        "start_time": (BASE + timedelta(hours=2)).isoformat(),
        "duration": 1800,
        "status": "completed",
        "profile_name": "Bob Express",
        "power_data": trace2,
        "energy_wh": 170.0,
    })

    await store.async_rebuild_envelope("Bob Express")
    env = store.get_envelope("Bob Express")

    assert env is not None
    assert env["cycle_count"] == 2
    # Real energy average is 170 Wh = 0.17 kWh
    assert env["avg_energy"] == pytest.approx(0.17, rel=1e-2)


@pytest.mark.asyncio
async def test_reference_only_profile_cycle_count_zero(store):
    """A profile seeded solely with reference cycles has cycle_count == 0."""
    offset_trace = [[float(i * 60), 100.0] for i in range(31)]
    await store.add_reference_cycle("Seeded Profile", offset_trace, {"store_cycle_id": "ref_1"})

    env = store.get_envelope("Seeded Profile")
    assert env is not None
    assert env["cycle_count"] == 0
    assert env["avg_energy"] is not None
    assert env["avg_energy"] > 0
