import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import pi

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "open-meteo-58.73N94.21W8m.csv"

meta = pd.read_csv(CSV_PATH, nrows=1)
ts = pd.read_csv(CSV_PATH, skiprows=3)
ts.columns = ts.columns.str.strip()

# Rename columns
rename_map = {
    "temperature_2m (°C)": "T_air_C",
    "temperature_2m (C)": "T_air_C",
    "apparent_temperature (°C)": "T_app_C",
    "apparent_temperature (C)": "T_app_C",
    "cloud_cover (%)": "cloud_pct",
    "relative_humidity_2m (%)": "RH_pct",
    "precipitation (mm)": "precip_mm",
}
ts = ts.rename(columns={k: v for k, v in rename_map.items() if k in ts.columns})

ts["time"] = pd.to_datetime(ts["time"])
ts = ts.sort_values("time").reset_index(drop=True)

LAT = float(meta.loc[0, "latitude"])
LON = float(meta.loc[0, "longitude"])

# solar position helper functions
def day_of_year(dt: pd.Series) -> np.ndarray:
    return dt.dt.dayofyear.to_numpy()

def solar_position(time: pd.Series, lat_deg: float, lon_deg: float) -> tuple[np.ndarray, np.ndarray]:

    n = day_of_year(time)
    hour = (time.dt.hour + time.dt.minute/60 + time.dt.second/3600).to_numpy()

    phi = np.deg2rad(lat_deg)

    # Earth's orbit is not perfectly circular and there is an axis tilt
    # This equation of time (EoT) converts between solar time and clock time
    B = 2*pi*(n - 81)/365.0
    eot_min = 9.87*np.sin(2*B) - 7.53*np.cos(B) - 1.5*np.sin(B)

    # Solar declination
    delta = np.deg2rad(23.45) * np.sin(2*pi*(284 + n)/365.0)

    # converts longitude degrees to hours as Earth rotates 15 degrees per hour
    # Hour angle tells us how far the sun is from solar noon (0 degrees at solar noon, +/- 15 degrees per hour)
    solar_time = hour + (lon_deg/15.0) + (eot_min/60.0)
    H = np.deg2rad(15.0*(solar_time - 12.0))

    # standard relationship between latitude (phi), declination (delta), hour hangle (H), and zenith angle (z)
    cosz = np.sin(phi)*np.sin(delta) + np.cos(phi)*np.cos(delta)*np.cos(H)
    cosz = np.clip(cosz, -1, 1)
    return cosz, delta, H

cosz, delta, H = solar_position(ts["time"], LAT, LON)
ts["cosz"] = cosz

# Global Horizontal Irradiance (GHI) gives us the total solar power per square meter on a flat horizontal surface
# Haurwitz clear-sky GHI:
# GHI_clear = 1098 * cosz * exp(-0.057 / cosz)
ghi_clear = np.where(
    cosz > 0,
    1098.0 * cosz * np.exp(-0.057 / np.clip(cosz, 1e-6, None)),
    0.0
)

# Cloud attenuation
# Kasten and Czeplak (1980) model: t_cloud = 1 - 0.75*(CC^3.4), where CC is cloud cover fraction (0 to 1)
CC = np.clip(ts["cloud_pct"].to_numpy()/100.0, 0, 1)
t_cloud = np.clip(1 - 0.75*(CC**3.4), 0.05, 1.0)

GHI = ghi_clear * t_cloud

ts["GHI_clear_Wm2"] = ghi_clear
ts["GHI_Wm2"] = GHI

# Erbs et al. (1982), using clearness index CI to split GHI into DHI and DNI components
n = day_of_year(ts["time"])
G_sc = 1367.0
G0n = G_sc * (1.0 + 0.033*np.cos(2*pi*n/365.0)) # extraterrestrial normal irradiance
G0h = G0n * cosz # extraterrestrial horizontal irradiance
# Source: https://www.fao.org/4/x0490e/x0490e07.htm

CI = np.where(G0h > 1e-6, GHI / G0h, 0.0)
CI = np.clip(CI, 0.0, 1.2)

# Calculate diffuse fraction Fd based on clearness index CI
Fd = np.zeros_like(CI)
m1 = CI <= 0.20 # cloudy
m2 = (CI > 0.20) & (CI <= 0.80) # partly cloudy
m3 = CI > 0.80 # clear sky
# equations for each category from Erbs et al. (1982)
Fd[m1] = 1.0 - 0.09*CI[m1]
Fd[m2] = (0.9511 - 0.1604*CI[m2] + 4.388*CI[m2]**2 - 16.638*CI[m2]**3 + 12.336*CI[m2]**4)
Fd[m3] = 0.165
Fd = np.clip(Fd, 0.0, 1.0)

DHI = Fd * GHI
beam_h = np.clip(GHI - DHI, 0, None)
DNI = np.where(cosz > 0, beam_h / np.clip(cosz, 1e-6, None), 0.0)

beta = np.deg2rad(abs(LAT))
albedo = 0.25

# POA irradiance on tilted panel
phi = np.deg2rad(LAT)
zen = np.arccos(np.clip(cosz, -1, 1)) # solar zenith angle
alpha = (pi/2) - zen # solar elevation angle

gamma_s = np.arctan2(np.cos(delta)*np.sin(H), np.sin(alpha)*np.sin(phi) - np.sin(delta))
gamma_p = np.deg2rad(180.0)

cos_theta_i = np.cos(zen)*np.cos(beta) + np.sin(zen)*np.sin(beta)*np.cos(gamma_s - gamma_p)
POA_beam = DNI * cos_theta_i

# Liu & Jordan isotropic diffuse model
# Used to estimate the diffuse sky on a tilted surface (assumes sky is uniform in composition)
# POA = Plane of Array
POA_diff = DHI * (1 + np.cos(beta)) / 2
POA_ground = albedo * GHI * (1 - np.cos(beta)) / 2
POA = POA_beam + POA_diff + POA_ground

ts["POA_Wm2"] = POA

# PV model (Si versus Tandem) with NOCT temperature
T_air = ts["T_air_C"].to_numpy()
NOCT = 45.0

T_cell = T_air + (POA/800.0)*(NOCT - 20.0)
ts["T_cell_C"] = T_cell
eta_inv = 0.97

def pv_power(POA_Wm2, T_cell_C, eta_stc, gamma):
    # efficiency at temperature
    eta = eta_stc * (1 + gamma*(T_cell_C - 25.0))
    eta = np.clip(eta, 0, None)
    P_ac = POA_Wm2 * eta * eta_inv
    return P_ac, eta

# Tech assumptions
P_si, eta_si = pv_power(POA, T_cell, eta_stc=0.21, gamma=-0.0045)
P_tn, eta_tn = pv_power(POA, T_cell, eta_stc=0.28, gamma=-0.0025)

ts["P_si_Wm2"] = P_si
ts["P_tandem_Wm2"] = P_tn
ts["eta_si_pct"] = 100*eta_si
ts["eta_tandem_pct"] = 100*eta_tn

# Energy per hour (kWh/m^2)
ts["E_si_kWh_m2"] = ts["P_si_Wm2"]/1000.0
ts["E_tandem_kWh_m2"] = ts["P_tandem_Wm2"]/1000.0

# monthly tables and size to Churchill's annual energy need (0.199 PJ/year)
ts["month"] = ts["time"].dt.to_period("M").astype(str)

monthly_energy = ts.groupby("month")[["E_si_kWh_m2","E_tandem_kWh_m2"]].sum()
monthly_eff = ts.groupby("month")[["eta_si_pct","eta_tandem_pct"]].mean()

# Annual energy need (0.199 PJ/year)
annual_kWh_need = 0.199e15 / 3.6e6

annual_kWh_per_m2_si = float(ts["E_si_kWh_m2"].sum())
annual_kWh_per_m2_tn = float(ts["E_tandem_kWh_m2"].sum())

A_needed_si = annual_kWh_need / annual_kWh_per_m2_si
A_needed_tn = annual_kWh_need / annual_kWh_per_m2_tn

summary = {
    "lat": LAT,
    "lon": LON,
    "rows": len(ts),
    "start": str(ts["time"].min()),
    "end": str(ts["time"].max()),
    "annual_need_kWh": annual_kWh_need,
    "annual_kWh_per_m2_silicon": annual_kWh_per_m2_si,
    "annual_kWh_per_m2_tandem": annual_kWh_per_m2_tn,
    "area_needed_m2_silicon": A_needed_si,
    "area_needed_km2_silicon": A_needed_si/1e6,
    "area_needed_m2_tandem": A_needed_tn,
    "area_needed_km2_tandem": A_needed_tn/1e6,
}

print("\nSUMMARY:")
for k, v in summary.items():
    print(f"{k:>28}: {v}")

# Plot weekly window of winter vs summer efficiency
def plot_eff_window(start, days, title):
    w = ts[(ts["time"] >= start) & (ts["time"] < start + pd.Timedelta(days=days))]
    plt.figure(figsize=(11, 4))
    plt.plot(w["time"], w["eta_si_pct"], label="Silicon (%)")
    plt.plot(w["time"], w["eta_tandem_pct"], label="Tandem (%)")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Efficiency (%)")
    plt.ylim(0, 35)
    plt.legend()
    plt.tight_layout()

plot_eff_window(pd.Timestamp("2025-01-10"), 7, "Winter sample: PV efficiency (%) - Churchill, Manitoba")
plot_eff_window(pd.Timestamp("2025-06-10"), 7, "Summer sample: PV efficiency (%) - Churchill, Manitoba")

# Plot monthly energy yield
plt.figure(figsize=(11, 4))
plt.plot(monthly_energy.index, monthly_energy["E_si_kWh_m2"], label="Silicon (kWh/m^2)")
plt.plot(monthly_energy.index, monthly_energy["E_tandem_kWh_m2"], label="Tandem (kWh/m^2)")
plt.title("Monthly PV energy yield per m^2 - Churchill, Manitoba")
plt.xlabel("Month")
plt.ylabel("kWh/m^2")
plt.xticks(rotation=60, ha="right")
plt.legend()
plt.tight_layout()

# Plot monthly average efficiency
plt.figure(figsize=(11, 4))
plt.plot(monthly_eff.index, monthly_eff["eta_si_pct"], label="Silicon avg (%)")
plt.plot(monthly_eff.index, monthly_eff["eta_tandem_pct"], label="Tandem avg (%)")
plt.title("Monthly average PV efficiency (%) - Churchill, Manitoba")
plt.xlabel("Month")
plt.ylabel("Efficiency (%)")
plt.xticks(rotation=60, ha="right")
plt.legend()
plt.tight_layout()

plt.show()