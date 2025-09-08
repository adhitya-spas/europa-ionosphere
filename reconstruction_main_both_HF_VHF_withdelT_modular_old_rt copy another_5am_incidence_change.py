# reconstruction_main_both_HF_VHF.py

# ------------------- MODULAR TOMOGRAPHY SCRIPT -------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from modify_df import load_mission_df
from ionosphere_design import build_ionosphere, add_gaussian_enhancement, plot_ionosphere
from ray_trace_passive import compute_STEC_along_path, dualfreq_to_VTEC, reconstruct_art, trace_passive_nadir, trace_passive_oblique
from improved_geometry import build_geometry_matrix_weighted, calculate_measurement_weights, weighted_reconstruction_art

def run_tomography(
    iono,
    lats,
    alts_m,
    ray_paths_vhf,
    ray_paths_hf,
    f_c_vhf,
    bw_vhf,
    integration_times_vhf,
    f_c_hf,
    bw_hf,
    integration_times_hf,
    theta_list_hf,
    n_iters=20,
    relax=0.1
):
    # VHF
    stec_vhf = np.array([
        compute_STEC_along_path(iono, lats, alts_m, ray) for ray in ray_paths_vhf
    ])
    vtec_vhf = []
    delta_t_vhf = []
    for i, ray_stec in enumerate(stec_vhf):
        vtec_i, dt_i = dualfreq_to_VTEC(
            stec_slant=np.array([ray_stec]),
            f_c=f_c_vhf,
            bw=bw_vhf,
            theta_deg=0.0,
            integration_time=integration_times_vhf[i],
            return_deltat=True
        )
        vtec_vhf.append(vtec_i[0])
        delta_t_vhf.append(dt_i[0])
    vtec_vhf = np.array(vtec_vhf)
    delta_t_vhf = np.array(delta_t_vhf)
    D_vhf = build_geometry_matrix_weighted(ray_paths_vhf, lats, alts_m)

    # HF
    all_vtec_hf = []
    all_deltat_hf = []
    for idx, ray in enumerate(ray_paths_hf):
        θ = theta_list_hf[idx]
        vtec, delta_t = dualfreq_to_VTEC(
            stec_slant=np.array([
                compute_STEC_along_path(iono, lats, alts_m, ray)
            ]),
            f_c=f_c_hf,
            bw=bw_hf,
            theta_deg=θ,
            integration_time=integration_times_hf[idx],
            return_deltat=True
        )
        all_vtec_hf.append(vtec)
        all_deltat_hf.append(delta_t)
    all_vtec_hf = np.hstack(all_vtec_hf)
    all_deltat_hf = np.hstack(all_deltat_hf)
    D_hf = build_geometry_matrix_weighted(ray_paths_hf, lats, alts_m)

    # ART inversions
    Ne_rec_vhf = reconstruct_art(D_vhf, vtec_vhf, len(lats), len(alts_m), n_iters, relax)
    Ne_rec_hf  = reconstruct_art(D_hf,  all_vtec_hf, len(lats), len(alts_m), n_iters, relax)
    D_all    = np.vstack([D_vhf, D_hf])
    vtec_all = np.concatenate([vtec_vhf, all_vtec_hf])
    Ne_rec_all = reconstruct_art(D_all, vtec_all, len(lats), len(alts_m), n_iters, relax)

    # δ(Δt)-based TEC Estimate Reconstruction
    origin_lat = ray_paths_hf[0][0,0]
    sat_lats = np.array([ray[0,0] for ray in ray_paths_vhf])
    origin_lat_idx = np.argmin(np.abs(sat_lats - origin_lat))
    delta_t_vhf_matched = np.full(len(all_deltat_hf), delta_t_vhf[origin_lat_idx])
    # Use HF frequency pair for freq_factor_h
    f1_h = f_c_hf - 0.5e6
    f2_h = f_c_hf + 0.5e6
    freq_factor_h = (f1_h**2 * f2_h**2) / (f2_h**2 - f1_h**2)
    c = 3e8
    delta_dt_all = -(delta_t_vhf_matched - all_deltat_hf)
    tec_est_ddt_all = (c * delta_dt_all / 80.6) * freq_factor_h
    Ne_rec_ddt = reconstruct_art(D_hf, tec_est_ddt_all, len(lats), len(alts_m), n_iters, relax)

    # Weighted reconstruction
    all_integration_times = np.concatenate([integration_times_vhf, integration_times_hf])
    all_bandwidths = np.concatenate([
        np.full(len(vtec_vhf), bw_vhf),
        np.full(len(all_vtec_hf), bw_hf)
    ])
    measurement_weights = calculate_measurement_weights(
        integration_times=all_integration_times,
        bandwidths=all_bandwidths
    )
    Ne_rec_weighted = weighted_reconstruction_art(
        D_all, vtec_all, measurement_weights, len(lats), len(alts_m), n_iters=50, relax=0.2
    )

    # Plot results
    fig, axs = plt.subplots(1, 5, figsize=(30, 6), sharey=True)
    titles = ["VHF only", "HF only", "Combined (standard)", "Combined (weighted)", "δ(Δt)-based"]
    recs = [Ne_rec_vhf, Ne_rec_hf, Ne_rec_all, Ne_rec_weighted, Ne_rec_ddt]
    for ax, rec, title in zip(axs, recs, titles):
        im = ax.pcolormesh(lats, alts_m/1e3, rec*1e-6, shading='auto', cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel("Latitude (°)")
        if ax is axs[0]:
            ax.set_ylabel("Altitude (km)")
        fig.colorbar(im, ax=ax, label='Ne (×10⁶ cm⁻³)')
    plt.tight_layout()
    plt.show(block=False)

    return {
        'Ne_rec_vhf': Ne_rec_vhf,
        'Ne_rec_hf': Ne_rec_hf,
        'Ne_rec_all': Ne_rec_all,
        'Ne_rec_weighted': Ne_rec_weighted,
        'Ne_rec_ddt': Ne_rec_ddt,
        'lats': lats,
        'alts_m': alts_m,
        'all_vtec_hf': all_vtec_hf,
        'all_deltat_hf': all_deltat_hf
    }


EUROPA_R_KM = 1569  # mean radius

def angular_resolution_deg(integration_time_s: float,
                           flyby_speed_kms: float = 5.0,
                           altitude_km: float = 1000.0) -> float:
    """
    Convert along-track distance traversed during an integration into
    an angular resolution (degrees) at altitude (R+h).
    s = v * T  (km),  
    Ang_res [rad] = s / (R+h),  
    Ang_res [deg] = Ang_res [rad] * 180/pi
    """
    s_km = flyby_speed_kms * integration_time_s
    return (s_km / (EUROPA_R_KM + altitude_km)) * (180.0 / np.pi)

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b)**2)))

def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cross-correlation for reconstruction quality ([-1,1])."""
    a0, b0 = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a0) * np.linalg.norm(b0)
    return float((a0 * b0).sum() / denom) if denom > 0 else 0.0

def run_recon_with_theta_cap(theta_cap_deg,
                             D_hf, ray_paths_hf, lats, alts_m,
                             vtec_hf_all, theta_per_ray_all,
                             n_iters=20, relax=0.1):
    sel = np.abs(90 - theta_per_ray_all) <= theta_cap_deg
    D_sub = D_hf[sel, :]
    vtec_sub = vtec_hf_all[sel]
    rec = reconstruct_art(D_sub, vtec_sub, len(lats), len(alts_m), n_iters, relax)
    return rec, sel.sum()

# Build a “ground truth” grid for scoring = your simulated iono (match orientation)
def ground_truth_grid(iono, lats, alts_m):
    # Your code uses rec shaped [n_lat x n_alt] or [n_alt x n_lat] depending on builder.
    # Align shapes: make both rec and truth (n_alt, n_lat)
    if iono.shape[0] == len(lats):
        truth = iono.T   # (alt, lat)
    else:
        truth = iono     # already (alt, lat)
    return truth

def centers_to_edges(vec):
    """Given a 1-D monotonic array of centers, return array of edges of length N+1."""
    vec = np.asarray(vec)
    d = np.diff(vec)
    edges = np.empty(vec.size + 1, dtype=vec.dtype)
    edges[1:-1] = vec[:-1] + d/2
    # For the two ends, extrapolate half a bin
    edges[0]  = vec[0]  - d[0]/2
    edges[-1] = vec[-1] + d[-1]/2
    return edges

# def as_alt_lat(grid, lats, alts_m):
#     # Want shape (n_alt, n_lat)
#     if grid.shape == (len(alts_m), len(lats)):
#         return grid
#     elif grid.shape == (len(lats), len(alts_m)):
#         return grid.T
#     else:
#         raise ValueError(f"Unexpected grid shape {grid.shape}")

# ------------------- EXAMPLE USAGE -------------------
if __name__ == "__main__":
    # Load or build ionosphere
    df = load_mission_df("new_mission_df.pkl")
    row = df[df["Mission"] == "E6a Exit"].iloc[0]
    alt_km_1d, Ne_1d = row["Altitude"], row["Ne"]
    lats, alt_km, iono = build_ionosphere(
        pd.DataFrame({"altitude": alt_km_1d, "ne": Ne_1d}),
        lat_extent=(-10, 10),
        lat_res=200
    )


    # Plot the electron density profile (ionogram) and the original ionosphere map as subplots
    center_lat_idx = np.argmin(np.abs(lats - 0.0))
    if iono.shape[0] == len(lats):
        Ne_profile = iono[center_lat_idx, :]
        iono_plot = iono
    else:
        Ne_profile = iono[:, center_lat_idx]
        iono_plot = iono.T
    fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    # Electron density profile
    axs[0].plot(Ne_profile, alt_km, 'b-', linewidth=2)
    axs[0].set_xlabel('Electron Density Ne (m$^{-3}$)')
    axs[0].set_ylabel('Altitude (km)')
    axs[0].set_title('Ne Profile at Center Latitude (0°)')
    axs[0].grid(True)
    # 2D ionosphere map

    edges_lats = centers_to_edges(lats)
    edges_alts = centers_to_edges(alt_km)
    im = axs[1].pcolormesh(edges_lats, edges_alts, iono_plot.T, shading='auto', cmap='viridis')
    # im = axs[1].pcolormesh(lats, alt_km, iono_plot.T, shading='auto', cmap='viridis')

    axs[1].set_xlabel('Latitude (°)')
    axs[1].set_title('Original Ionosphere (Ne)')
    fig.colorbar(im, ax=axs[1], label='Ne (m$^{-3}$)')
    plt.tight_layout()
    plt.show(block=False)



    iono = add_gaussian_enhancement(
        ionosphere_map=iono,
        latitudes=lats,
        altitude=alt_km,
        lat_center=0.0,
        alt_center=150.0,
        lat_width=1.0,
        alt_width=20.0,
        amplitude=5e11
    )
    alts_m = alt_km * 1e3

        # Plot the electron density profile (ionogram) and the original ionosphere map as subplots
    center_lat_idx = np.argmin(np.abs(lats - 0.0))
    if iono.shape[0] == len(lats):
        Ne_profile = iono[center_lat_idx, :]
        iono_plot = iono
    else:
        Ne_profile = iono[:, center_lat_idx]
        iono_plot = iono.T
    fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    # Electron density profile
    axs[0].plot(Ne_profile, alt_km, 'b-', linewidth=2)
    axs[0].set_xlabel('Electron Density Ne (m$^{-3}$)')
    axs[0].set_ylabel('Altitude (km)')
    axs[0].set_title('Ne Profile at Center Latitude (0°)')
    axs[0].grid(True)
    # 2D ionosphere map
    edges_lats = centers_to_edges(lats)
    edges_alts = centers_to_edges(alt_km)
    im = axs[1].pcolormesh(edges_lats, edges_alts, iono_plot.T, shading='auto', cmap='viridis')
    # im = axs[1].pcolormesh(lats, alt_km, iono_plot.T, shading='auto', cmap='viridis')
    axs[1].set_xlabel('Latitude (°)')
    axs[1].set_title('Original Ionosphere (Ne)')
    fig.colorbar(im, ax=axs[1], label='Ne (m$^{-3}$)')
    plt.tight_layout()
    plt.show(block=False)

    # User: define your ray paths and parameters here!
    # Example: VHF nadir rays
    from ray_trace_passive import trace_passive_nadir
    sat_lats = np.linspace(-6.0, 6.0, 60)
    rays_vhf = trace_passive_nadir(sat_lats, alts_m, npts=500)
    integration_times_vhf = np.full(len(rays_vhf), 0.12)
    f_c_vhf = 60e6
    bw_vhf = 10e6



    # HF oblique rays: match copy 3 logic
    # Define angles as angle from horizontal (original convention)
    incidence_angles = [60, 50, 40, 30, 20, 10, 5]  # Modify TODO
    theta_list = [30, 40, 50, 60, 70, 80, 85]  # angle from horizontal (legacy, do not use for calculations)
    
    integration_times_hf_per_angle = np.array([0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14])
    # integration_times_hf_per_angle = np.array([0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08])
    # Compute HF integration time per angle using provided formula:
    # T(theta) = 1500 * sin(theta) * (180 - 2*theta) * (pi/180), with theta in degrees
    theta_arr = np.array(theta_list, dtype=float)
    ang_vel_spacecraft = 0.00872664625 # km/s
    # integration_times_hf_per_angle = ((np.sin(np.deg2rad(theta_arr))/1500) * np.deg2rad(180.0 - (2.0 * theta_arr))) / ang_vel_spacecraft

    f_c_hf = 9e6
    bw_hf = 1e6

    all_rays_hf = []
    all_vtec_hf = []
    all_deltat_hf = []
    integration_times_hf = []
    theta_per_ray = []

    for idx, θ in enumerate(theta_list):
        T_hf = integration_times_hf_per_angle[idx]
        rays_hf = trace_passive_oblique(
            sat_lats=sat_lats,
            h0=alts_m.max(),
            hsat=alts_m.max(),
            theta_i_deg=θ,
            npts=500
        )
        stec = np.array([
            compute_STEC_along_path(iono, lats, alts_m, ray)
            for ray in rays_hf
        ])
        vtec, delta_t = dualfreq_to_VTEC(
            stec_slant=stec,
            f_c=f_c_hf,
            bw=bw_hf,
            theta_deg=θ,
            integration_time=T_hf,
            return_deltat=True
        )
        all_rays_hf.extend(rays_hf)
        all_vtec_hf.append(vtec)
        all_deltat_hf.append(delta_t)
        integration_times_hf.extend([T_hf] * len(rays_hf))
        theta_per_ray.extend([θ] * len(rays_hf))

    all_vtec_hf = np.hstack(all_vtec_hf)
    all_deltat_hf = np.hstack(all_deltat_hf)
    integration_times_hf = np.array(integration_times_hf)
    theta_per_ray = np.array(theta_per_ray)
    rays_hf = all_rays_hf

    altitude_km_for_ang = alts_m.max() / 1e3 
    ang_res_deg_per_ray = np.array([
        angular_resolution_deg(T, flyby_speed_kms=5.0, altitude_km=altitude_km_for_ang) # Need to change flyby_speed_kms TODO
        for T in integration_times_hf
    ])


    # Plot HF and VHF raypaths side by side as subplots, color HF by true incidence angle (from vertical)
    fig, axs = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    # Compute true incidence angles for legend and coloring
    unique_thetas = np.unique(theta_list)
    unique_incidence_angles = 90 - unique_thetas
    cmap = cm.get_cmap('plasma', len(unique_incidence_angles))
    color_dict = {inc_angle: cmap(i) for i, inc_angle in enumerate(unique_incidence_angles)}

    # For each ray, use the true incidence angle for color
    ray_idx = 0
    for idx, theta in enumerate(theta_list):
        incidence_angle = 90 - theta
        n_rays = sum(np.array(theta_per_ray) == theta)
        for _ in range(n_rays):
            ray = rays_hf[ray_idx]
            lat = ray[:, 0]
            alt_km = ray[:, 1] / 1e3
            axs[0].plot(lat, alt_km, color=color_dict[incidence_angle], linewidth=0.5, alpha=0.7)
            ray_idx += 1
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=color_dict[inc_angle], lw=2, label=f'{inc_angle}°') for inc_angle in unique_incidence_angles]
    axs[0].legend(handles=legend_elements, title='HF θ (deg, from vertical)', loc='upper right', fontsize='small')
    axs[0].set_xlabel('Latitude (Degree)')
    axs[0].set_ylabel('Altitude (km)')
    axs[0].set_title('Modeled Raypaths (HF)')
    axs[0].set_xlim(np.min(sat_lats)-2, np.max(sat_lats)+2)
    axs[0].set_ylim(0, 1000)
    # VHF raypaths (all blue)
    for ray in rays_vhf:
        lat = ray[:, 0]
        alt_km = ray[:, 1] / 1e3
        axs[1].plot(lat, alt_km, color='b', linewidth=0.5, alpha=0.5)
    axs[1].set_xlabel('Latitude (Degree)')
    axs[1].set_title('Modeled Raypaths (VHF)')
    axs[1].set_xlim(np.min(sat_lats)-2, np.max(sat_lats)+2)
    axs[1].set_ylim(0, 1000)
    plt.tight_layout()
    plt.show(block=False)

    # Run the tomography pipeline

    results = run_tomography(
        iono=iono,
        lats=lats,
        alts_m=alts_m,
        ray_paths_vhf=rays_vhf,
        ray_paths_hf=rays_hf,
        f_c_vhf=f_c_vhf,
        bw_vhf=bw_vhf,
        integration_times_vhf=integration_times_vhf,
        f_c_hf=f_c_hf,
        bw_hf=bw_hf,
        integration_times_hf=integration_times_hf,
        theta_list_hf=theta_per_ray,
        n_iters=20,
        relax=0.1
    )



# After tomography run, build DataFrame and plot error using returned arrays




# --- Compute noise-free truth and error for each HF ray ---
stec_true = np.array([compute_STEC_along_path(iono, lats, alts_m, ray) for ray in rays_hf])
vtec_true = stec_true * np.cos(np.deg2rad(theta_per_ray))  # verticalize the truth
vtec_noisy  = results['all_vtec_hf']        # already verticalized inside dualfreq_to_VTEC
t_meas      = integration_times_hf.copy()   # one T per ray
tec_for_bin = vtec_true                     # bin by true VTEC (like the paper)
delta_tec_err = np.abs(vtec_noisy - vtec_true)

# --- plot exactly as before but using the corrected quantities ---
df = pd.DataFrame({"vtec": tec_for_bin, "t": t_meas, "err": delta_tec_err})
df = df[(df['t'] >= 0.05) & (df['t'] <= 0.15)]
df = df[(df['vtec'] >= 1.5e15) & (df['vtec'] <= 5e15)]

tec_bins = np.linspace(1.5e15, 5e15, 50)
t_bins   = np.linspace(0.05, 0.15, 50)

heatmap, yedges, xedges = np.histogram2d(df['vtec'], df['t'], bins=[tec_bins, t_bins], weights=df['err'])
counts,  _,     _       = np.histogram2d(df['vtec'], df['t'], bins=[tec_bins, t_bins])
avg_error = heatmap / np.maximum(counts, 1)

plt.figure(figsize=(8,6))
plt.pcolormesh(xedges, yedges, avg_error, shading='auto', cmap='plasma')
plt.xlabel("Integration Time (s)")
plt.ylabel("TEC (electrons/m²)")
plt.title("Average TEC Estimate Error for Passive (Simulated)")
plt.colorbar(label="ΔTEC Error (m⁻²)")
plt.tight_layout(); plt.show(block=False)

# Filter time range matching Peters et al. (Fig. 7)
df = df[(df['t'] >= 0.05) & (df['t'] <= 0.15)]
#df = df[df['vtec'] <= 5e15]  # Match TEC range for meaningful comparison

# Bin edges
tec_bins = np.linspace(1.5e15, 5e15, 50)
t_bins = np.linspace(0.05, 0.15, 50)

# 2D histogram with averaging
heatmap, yedges, xedges = np.histogram2d(df['vtec'], df['t'], bins=[tec_bins, t_bins], weights=df['err'])
counts, _, _ = np.histogram2d(df['vtec'], df['t'], bins=[tec_bins, t_bins])
avg_error = heatmap / np.maximum(counts, 1)

# Plot
plt.figure(figsize=(8, 6))
plt.pcolormesh(xedges, yedges, avg_error, shading='auto', cmap='plasma')
plt.xlabel("Integration Time (s)")
plt.ylabel("TEC (electrons/m²)")
plt.title("Average TEC Estimate Error for Passive (Simulated)")
plt.colorbar(label="ΔTEC Error (m⁻²)")
plt.tight_layout()
plt.show(block=False)

# --------------------------------------------------------------------------
# VHF Range Error after Passive HF Correction (2D map)
# --------------------------------------------------------------------------
f_vhf = 60e6  # Hz

# Use delta_tec_err (already in electrons/m^2) as the TEC error
# Convert to range error: Δr = 40.3 * ΔTEC / f^2 (in meters)
df_range = df.copy()
df_range["range_err"] = 40.3 * df_range["err"] / f_vhf**2  # meters

# 2D histogram with averaging (same bins as before)
heatmap_r, yedges_r, xedges_r = np.histogram2d(
    df_range['vtec'], df_range['t'], bins=[tec_bins, t_bins], weights=df_range['range_err'])
counts_r, _, _ = np.histogram2d(
    df_range['vtec'], df_range['t'], bins=[tec_bins, t_bins])
avg_range_err = heatmap_r / np.maximum(counts_r, 1)

# Plot
plt.figure(figsize=(8, 6))
plt.pcolormesh(xedges_r, yedges_r, avg_range_err, shading='auto', cmap='jet')
plt.xlabel("Integration Time (sec)")
plt.ylabel("Total Electron Content (m$^{-2}$)")
plt.title("VHF Range Error after Passive HF Correction")
cbar = plt.colorbar(label="Δr (meters)")
plt.tight_layout()
plt.show(block=False)


# --- TEC error arrays you already have ---
# vtec_true, vtec_noisy, integration_times_hf, theta_per_ray
# scatter of |ΔTEC| vs T with a twin x‑axis showing angular resolution (your Step‑1 idea)
# scatter of |ΔTEC| vs θ
# a bar chart to compare θ‑coverage scenarios (±20/40/60°)

tec_err = np.abs(vtec_noisy - vtec_true)

# (A) Error vs integration time (with angular resolution annotation)
import matplotlib.pyplot as plt
plt.figure(figsize=(7,5))
plt.scatter(integration_times_hf, tec_err, s=8, alpha=0.3)
plt.xlabel("Integration time T (s)")
plt.ylabel("|ΔTEC| (m$^{-2}$)")
plt.title("TEC Error vs Integration Time")
# annotate a second axis with angular resolution scale (0.1 s ≈ ~0.011° at 1000 km)
ax = plt.gca()
ax2 = ax.twiny()
xt = np.linspace(integration_times_hf.min(), integration_times_hf.max(), 5)
ax2.set_xticks(xt)
ax2.set_xlim(ax.get_xlim())
ax2.set_xticklabels([f"{angular_resolution_deg(x, 5.0, altitude_km_for_ang):.3f}°" for x in xt])
ax2.set_xlabel("Approx. angular resolution at altitude (deg)")
plt.tight_layout(); plt.show(block=False)

# (B) Error vs incidence angle

plt.figure(figsize=(7,5))
plt.scatter(90 - theta_per_ray, tec_err, s=8, alpha=0.3)
plt.xlabel("Incidence angle θ (deg, from vertical)")
plt.ylabel("|ΔTEC| (m$^{-2}$)")
plt.title("TEC Error vs θ (from vertical)")
plt.tight_layout(); plt.show(block=False)

# (C) Compare θ-limited “situations” (±20°, ±40°, ±60°)
# def summarize_theta_cap(cap):
#     sel = np.abs(theta_per_ray) <= cap
#     return cap, np.median(tec_err[sel]), np.quantile(tec_err[sel], [0.25, 0.75])
def summarize_theta_cap(cap):
    sel = np.abs(90 - theta_per_ray) <= cap
    if not np.any(sel):
        # No data for this cap — return NaNs so plotting can skip it
        return cap, np.nan, (np.nan, np.nan)
    vals = tec_err[sel]
    return (
        cap,
        float(np.nanmedian(vals)),
        (float(np.nanquantile(vals, 0.25)), float(np.nanquantile(vals, 0.75)))
    )

caps = [20, 40, 60]
summary = [summarize_theta_cap(c) for c in caps]
summary = [s for s in summary if np.isfinite(s[1])]

plt.figure(figsize=(6,4))
plt.bar([str(s[0]) for s in summary], [s[1] for s in summary])
plt.ylabel("Median |ΔTEC| (m$^{-2}$)")
plt.xlabel("θ cap (deg)")
plt.title("TEC Error vs θ-coverage")
plt.tight_layout(); plt.show(block=False)


# anotehr addition
# Overlay binned mean/std on the TEC–T heatmap (optional)
t_bins_line = np.linspace(0.05, 0.15, 16)
bin_idx = np.digitize(integration_times_hf, t_bins_line)
means = [tec_err[bin_idx==i].mean() if np.any(bin_idx==i) else np.nan for i in range(1, len(t_bins_line))]
plt.figure(figsize=(7,5))
plt.plot(0.5*(t_bins_line[:-1]+t_bins_line[1:]), means, marker='o')
plt.xlabel("Integration Time (s)")
plt.ylabel("Mean |ΔTEC| (m$^{-2}$)")
plt.title("Binned mean TEC error vs T")
plt.tight_layout(); plt.show(block=False)


# Inversion‑vs‑θ study (quality of ART recon vs angle)
D_hf = build_geometry_matrix_weighted(rays_hf, lats, alts_m)
truth = ground_truth_grid(iono, lats, alts_m)

theta_caps = [20, 40, 60, 80]
scores = []
for cap in theta_caps:
    rec_cap, n_used = run_recon_with_theta_cap(
        cap, D_hf, rays_hf, lats, alts_m, results['all_vtec_hf'], theta_per_ray,
        n_iters=20, relax=0.1
    )
    # Orient rec to (alt, lat) for scoring
    rec_cap_altlat = rec_cap
    if rec_cap_altlat.shape != truth.shape:
        rec_cap_altlat = rec_cap_altlat.T
    scores.append({
        "cap": cap,
        "n_rays": n_used,
        "rmse": rmse(rec_cap_altlat, truth),
        "ncc": ncc(rec_cap_altlat, truth)
    })

print("\nInversion vs θ results:")
for s in scores:
    print(f"θ≤{s['cap']:2d}°  rays={s['n_rays']:4d}  RMSE={s['rmse']:.3e}  NCC={s['ncc']:.3f}")

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(7,5))
plt.plot([s["cap"] for s in scores], [s["rmse"] for s in scores], marker='o')
plt.xlabel("θ cap (deg)")
plt.ylabel("Reconstruction RMSE (Ne units)")
plt.title("Inversion quality vs θ coverage (ART)")
plt.tight_layout(); plt.show(block=False)

plt.figure(figsize=(7,5))
plt.plot([s["cap"] for s in scores], [s["ncc"] for s in scores], marker='o')
plt.xlabel("θ cap (deg)")
plt.ylabel("Reconstruction NCC (–1…1)")
plt.title("Inversion quality vs θ coverage (ART)")
plt.tight_layout(); plt.show(block=False)

input("Press Enter to exit…")