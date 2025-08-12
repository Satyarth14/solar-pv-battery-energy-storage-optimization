"""
solar_pv_simulation.py
Simple solar PV simulation (hourly) using pvlib clearsky + PVWatts-like approximation.

Outputs:
 - hourly CSV with irradiance and AC power (kW)
 - plots: power vs time, cumulative energy
"""
import pvlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def get_float(prompt, default=None, min_val=None, max_val=None):
    while True:
        try:
            s = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
            if s == "" and default is not None:
                return float(default)
            v = float(s)
            if min_val is not None and v < min_val:
                print(f"Value must be >= {min_val}")
                continue
            if max_val is not None and v > max_val:
                print(f"Value must be <= {max_val}")
                continue
            return v
        except ValueError:
            print("Enter a numeric value.")

def get_str(prompt, default=None):
    s = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
    return s if s else default

def main():
    print("Simple Solar PV Simulation (hourly) — PVLib + PVWatts-style approximation\n")

    # --- Inputs ---
    lat = get_float("Latitude (deg)", default=22.57)               # default: New Delhi ~22.57
    lon = get_float("Longitude (deg)", default=88.36)             # default: Kolkata ~88.36 (user can change)
    tz = get_str("Timezone (tz database string)", default="Asia/Kolkata")
    start_date = get_str("Start date (YYYY-MM-DD)", default=datetime.now().strftime("%Y-%m-%d"))
    end_date = get_str("End date (YYYY-MM-DD)", default=start_date)
    system_kw = get_float("System capacity (kW STC)", default=1.0, min_val=0.01)
    tilt = get_float("Tilt angle (deg, 0=flat)", default=None)
    if tilt is None:
        # rule-of-thumb: tilt ~ latitude
        tilt = round(lat, 1)
    azimuth = get_float("Azimuth (deg, 180 = south in northern hemisphere)", default=180)
    derate = get_float("System derate / performance ratio (0-1)", default=0.82, min_val=0.5, max_val=1.0)
    freq = get_str("Time resolution (H for hourly, 30T for 30-min)", default="H")

    # --- Build time index ---
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)  # include full last day
    times = pd.date_range(start=start, end=end, freq=freq, tz=tz)

    # --- Location and solar position + clearsky ---
    site = pvlib.location.Location(latitude=lat, longitude=lon, tz=tz)
    solpos = site.get_solarposition(times)
    # clearsky model (ineichen is reasonable and doesn't require external data)
    cs = site.get_clearsky(times, model="ineichen")  # returns ghi, dni, dhi

    # --- Plane-of-array irradiance (fixed tilt) ---
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solpos['apparent_zenith'],
        solar_azimuth=solpos['azimuth'],
        dni=cs['dni'],
        ghi=cs['ghi'],
        dhi=cs['dhi']
    )
    poa_global = poa['poa_global'].clip(lower=0)

    # --- Simple DC/AC power estimate (PVWatts-like) ---
    # P_dc ≈ system_kw * (poa_global / 1000)
    p_dc_kw = system_kw * (poa_global / 1000.0)
    p_ac_kw = p_dc_kw * derate              # simple derate to approximate inverter & system losses
    # zero-out negative values
    p_ac_kw = p_ac_kw.clip(lower=0)

    # --- Package results ---
    df = pd.DataFrame({
        "ghi_wm2": cs['ghi'],
        "dni_wm2": cs['dni'],
        "dhi_wm2": cs['dhi'],
        "poa_wm2": poa_global,
        "p_dc_kw": p_dc_kw,
        "p_ac_kw": p_ac_kw
    }, index=times)

    # --- Summary ---
    hours = (df.index[1] - df.index[0]).total_seconds() / 3600.0
    total_energy_kwh = df["p_ac_kw"].sum() * hours
    print("\nSimulation complete.")
    print(f"Period: {start_date} to {end_date} ({len(df)} points)")
    print(f"System capacity: {system_kw} kW  |  Tilt: {tilt}°, Azimuth: {azimuth}°")
    print(f"Estimated energy: {total_energy_kwh:.2f} kWh")

    # --- Save CSV ---
    out_csv = f"pv_results_{start_date}_to_{end_date}.csv"
    df.to_csv(out_csv, index_label="time")
    print(f"Hourly results saved: {out_csv}")

    # --- Plot: Power vs Time ---
    plt.figure(figsize=(10,4))
    plt.plot(df.index, df["p_ac_kw"])
    plt.ylabel("AC Power (kW)")
    plt.title(f"PV Power profile ({start_date} to {end_date})")
    plt.tight_layout()
    fname1 = "pv_power_profile.png"
    plt.savefig(fname1, dpi=200)
    plt.show()
    print(f"Saved plot: {fname1}")

    # --- Plot: Cumulative energy (kWh) ---
    cum_energy = df["p_ac_kw"].cumsum() * hours  # kWh cumulative
    plt.figure(figsize=(10,4))
    plt.plot(df.index, cum_energy)
    plt.ylabel("Cumulative energy (kWh)")
    plt.title("Cumulative energy generated")
    plt.tight_layout()
    fname2 = "pv_cumulative_energy.png"
    plt.savefig(fname2, dpi=200)
    plt.show()
    print(f"Saved plot: {fname2}")

    # Write a short summary file
    summary = {
        "start_date": start_date, "end_date": end_date,
        "lat": lat, "lon": lon, "tz": tz,
        "system_kw": system_kw, "tilt": tilt, "azimuth": azimuth,
        "derate": derate, "total_kwh": total_energy_kwh
    }
    pd.Series(summary).to_csv("pv_summary.csv")
    print("Saved summary: pv_summary.csv")

if __name__ == "__main__":
    main()
