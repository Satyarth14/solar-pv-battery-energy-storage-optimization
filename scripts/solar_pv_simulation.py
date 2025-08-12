import pvlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import cvxpy as cp

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

    lat = get_float("Latitude (deg)", default=22.57)
    lon = get_float("Longitude (deg)", default=88.36)
    tz = get_str("Timezone (tz database string)", default="Asia/Kolkata")
    start_date = get_str("Start date (YYYY-MM-DD)", default=datetime.now().strftime("%Y-%m-%d"))
    end_date = get_str("End date (YYYY-MM-DD)", default=start_date)
    system_kw = get_float("System capacity (kW STC)", default=1.0, min_val=0.01)
    tilt = get_float("Tilt angle (deg, 0=flat)", default=None)
    if tilt is None:
        tilt = round(lat, 1)
    azimuth = get_float("Azimuth (deg, 180 = south in northern hemisphere)", default=180)
    derate = get_float("System derate / performance ratio (0-1)", default=0.82, min_val=0.5, max_val=1.0)
    freq = get_str("Time resolution (H for hourly, 30T for 30-min)", default="H")

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    times = pd.date_range(start=start, end=end, freq=freq, tz=tz)

    site = pvlib.location.Location(latitude=lat, longitude=lon, tz=tz)
    solpos = site.get_solarposition(times)
    cs = site.get_clearsky(times, model="ineichen")

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

    p_dc_kw = system_kw * (poa_global / 1000.0)
    p_ac_kw = p_dc_kw * derate
    p_ac_kw = p_ac_kw.clip(lower=0)

    df = pd.DataFrame({
        "ghi_wm2": cs['ghi'],
        "dni_wm2": cs['dni'],
        "dhi_wm2": cs['dhi'],
        "poa_wm2": poa_global,
        "p_dc_kw": p_dc_kw,
        "p_ac_kw": p_ac_kw
    }, index=times)

    hours = (df.index[1] - df.index[0]).total_seconds() / 3600.0
    total_energy_kwh = df["p_ac_kw"].sum() * hours
    print(f"\nPV Simulation complete. Estimated energy: {total_energy_kwh:.2f} kWh")

    out_csv = f"pv_results_{start_date}_to_{end_date}.csv"
    df.to_csv(out_csv, index_label="time")
    print(f"Saved PV results: {out_csv}")

    plt.figure(figsize=(10,4))
    plt.plot(df.index, df["p_ac_kw"])
    plt.ylabel("AC Power (kW)")
    plt.title(f"PV Power profile ({start_date} to {end_date})")
    plt.tight_layout()
    plt.savefig("pv_power_profile.png", dpi=200)
    plt.show()

    hours_of_day = df.index.hour
    load_kw = 1.5 + 0.5 * (np.sin((hours_of_day - 7) / 12 * np.pi) + 1) \
        + 1.5 * (np.exp(-0.5*((hours_of_day-8)/1.5)**2) + np.exp(-0.5*((hours_of_day-19)/1.5)**2))
    load_kw = np.maximum(load_kw, 0.5)
    df['load_kW'] = load_kw

    print("\nRunning battery storage optimization...")
    T = len(df)
    pv = df['p_ac_kw'].values
    load = df['load_kW'].values
    tariff = 0.12 + 0.18 * ((df.index.hour >= 17) & (df.index.hour <= 21)).astype(float)  # $/kWh

    cap_kwh = 8.0
    p_charge_max = 3.0
    p_discharge_max = 3.0
    eta_charge = 0.97
    eta_discharge = 0.97
    soc_init = 0.5 * cap_kwh
    soc_min = 0.05 * cap_kwh
    soc_max = cap_kwh
    dt = 1.0

    charge = cp.Variable(T, nonneg=True)
    discharge = cp.Variable(T, nonneg=True)
    soc = cp.Variable(T+1)
    grid_import = cp.Variable(T, nonneg=True)

    constraints = [soc[0] == soc_init]
    for t in range(T):
        constraints += [
            soc[t+1] == soc[t] + (charge[t] * eta_charge - discharge[t] / eta_discharge) * dt,
            soc[t+1] >= soc_min,
            soc[t+1] <= soc_max,
            charge[t] <= p_charge_max,
            discharge[t] <= p_discharge_max,
            grid_import[t] >= load[t] - pv[t] - discharge[t] + charge[t]
        ]

    objective = cp.Minimize(cp.sum(cp.multiply(grid_import, tariff)))
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)

    df['charge_kW'] = charge.value
    df['discharge_kW'] = discharge.value
    df['SoC_kWh'] = soc.value[1:]
    df['grid_import_kW'] = grid_import.value
    df['tariff'] = tariff

    baseline_import = np.maximum(load - pv, 0)
    baseline_cost = np.sum(baseline_import * tariff)
    optimized_cost = np.sum(df['grid_import_kW'] * df['tariff'])
    print(f"Baseline cost: ${baseline_cost:.2f}")
    print(f"Optimized cost: ${optimized_cost:.2f}")
    print(f"Cost savings: ${baseline_cost - optimized_cost:.2f}")

    plt.figure(figsize=(12,8))
    plt.subplot(3,1,1)
    plt.plot(df.index, df['load_kW'], label='Load')
    plt.plot(df.index, df['p_ac_kw'], label='PV')
    plt.plot(df.index, df['grid_import_kW'], label='Grid Import')
    plt.legend(); plt.title('Power Profiles')

    plt.subplot(3,1,2)
    plt.plot(df.index, df['charge_kW'], label='Charge')
    plt.plot(df.index, df['discharge_kW'], label='Discharge')
    plt.legend(); plt.title('Battery Power')

    plt.subplot(3,1,3)
    plt.plot(df.index, df['SoC_kWh'], label='State of Charge')
    plt.bar(df.index, df['tariff'], alpha=0.3, label='Tariff ($/kWh)')
    plt.legend(); plt.title('SoC & Tariff')

    plt.tight_layout()
    plt.savefig("pv_battery_optimization.png", dpi=150)
    plt.show()

    df.to_csv("pv_battery_results.csv")
    print("Saved combined PV + battery results: pv_battery_results.csv")

if __name__ == "__main__":
    main()
