# Solar PV + Battery Energy Storage Optimization

This project simulates solar photovoltaic (PV) generation using **PVLib** and optimizes the charging/discharging schedule of a battery energy storage system using **convex optimization (CVXPy)**.

The goal is to reduce electricity costs under a **time-of-use (TOU) tariff** while meeting load demand, demonstrating **renewable integration and smart energy management**.

---

## Features
- **PV Simulation**:
  - Uses PVLib with the Ineichen clear-sky model.
  - Configurable location, tilt, azimuth, and system size.

- **Battery Optimization**:
  - Linear programming formulation with State of Charge (SoC), charge/discharge limits, and efficiency.
  - Time-of-use tariff integration.
  - Cost savings calculation vs. baseline (no battery).

- **Visualization**:
  - PV generation profile.
  - Battery charge/discharge power.
  - SoC with tariff overlay.
  - Grid import comparison (baseline vs optimized).

---

## Sample Results
Example run (1 kW PV, 8 kWh battery, New Delhi, hourly resolution):

- **Baseline Cost**: $X.XX/day
- **Optimized Cost**: $Y.YY/day
- **Savings**: $Z.ZZ/day (~P% reduction)


---

## Requirements
```bash
pip install -r requirements.txt
```
or (Anaconda):
```bash
conda install -c conda-forge pvlib cvxpy matplotlib pandas numpy scs
```

---

## Usage
```bash
python solar_pv_simulation.py
```
You will be prompted for:
- Location (latitude, longitude, timezone)
- System parameters (capacity, tilt, azimuth, derate)
- Simulation date range

---

## Project Structure
```
solar_pv_simulation.py       # Main simulation + optimization script
requirements.txt             # Dependencies
README.md                    # Project documentation
pv_battery_results.csv       # Sample output data
pv_battery_optimization.png  # Sample plot
```

---

## Future Improvements
- Use real residential/commercial load profiles from open datasets.
- Add weather-based PV forecasting.
- Implement grid export with feed-in tariffs.
