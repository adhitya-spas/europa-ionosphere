# Simulating a Combined Active-Passive, Dual-Frequency Radar Reconstruction of Europa's Ionospheric Profile

## Project Overview

This repository demonstrates a passive radar remote sensing methodology for ionospheric reconstruction, making it highly cost-effective and broadly applicable across planetary missions. The approach leverages existing spacecraft radio capabilities to perform dual-frequency sounding experiments, reducing mission complexity and enabling adaptation to current orbiters.

The methodology reconstructs Europa's ionospheric electron density using historical in situ profiles from Galileo and Juno [Kliore et al., 1997; Parisi et al., 2023], and builds on passive dual-frequency radio sounding techniques [Peters et al., 2020]. Profiles are optimized to eliminate non-physical negative Total Electron Content artifacts, creating clean day/night ionospheric models. Realistic variability is introduced through Gaussian-distributed transient enhancements modeling ionospheric disturbances across latitude-altitude grids.

Both nadir (VHF, 60 MHz ± 5 MHz) and oblique (HF, 9 MHz ± 0.5 MHz) signals are modeled as Jovian burst plane waves reflecting off Europa's surface and recorded by a low-altitude orbiter. Advanced radar signal processing converts slant TEC measurements along vertical and two-leg ray paths to Vertical TEC using dual-frequency group-delay formulations. An Algebraic Reconstruction Technique (ART) geometry matrix is populated for distinct cases: VHF only, HF only, combined dual-frequency approach, and reference-based radar correction approach, to invert for 2-D electron density reconstructions [Cushley et al., 2020].

This technology is universally applicable to any planet or moon with an ionosphere, eliminating instrument mass, power, and complexity penalties associated with active transmitters. Its compatibility with existing spacecraft radio systems enables immediate implementation on current and planned missions, offering mapping capabilities while minimizing operational risk and ground support requirements. This scalable, cost-effective approach could transform routine spacecraft communications and receivers into powerful scientific instruments for ionospheric reconstruction.

## Plain Language Summary

Jupiter's moon Europa has a thin layer of electrically charged particles called an ionosphere. These charged particles are plotted to create electron density profiles. Scientists have very limited knowledge of Europa's ionosphere from a couple of spacecraft flybys. Using data from those missions, we create cleaner day and night profiles of the ionosphere for our simulation.

The simulation considers two frequencies: one from natural radio bursts from Jupiter and another from the spacecraft's own system. By comparing how both signals change as they travel through the ionosphere, we can create detailed maps showing where the charged particles are located, without needing a separate instrument to do the same task.

This research proves the method works and can be used with any future spacecraft visiting Europa (or other planets with similar atmospheres) as long as radio equipment is onboard, which is standard on almost all spacecraft nowadays. This approach makes studying planetary atmospheres cheaper and uses fewer resources.

## References

- Kliore, A. J. et al. (1997). The Ionosphere of Europa from Galileo Radio Occultations. Science, 277, 355-358. https://doi.org/10.1126/science.277.5324.355
- Cushley, A. C., & Noel, J.-M. (2020). Ionospheric sounding and tomography using Automatic Identification System (AIS) and other signals of opportunity. Radio Science, 55, e2019RS006872. https://doi.org/10.1029/2019RS006872
- Parisi, M., Caruso, A., Buccino, D. R., Gramigna, E., Withers, P., Gomez-Casajus, L., et al. (2023). Radio occultation measurements of Europa's ionosphere from Juno's close flyby. Geophysical Research Letters, 50, e2023GL106637. https://doi.org/10.1029/2023GL106637
- Peters, S., Schroeder, D., & Romero-Wolf, A. (2020). Passive radio sounding to correct for Europa's ionospheric distortion of VHF signals. Planetary and Space Science, 187, 104925. https://doi.org/10.1016/j.pss.2020.104925

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd Ionosphere_Tomography
   ```

2. **Create a Python environment (recommended):**
   ```sh
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install required packages:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the main simulation:**
   ```sh
   python reconstruction_main_both_HF_VHF_withdelT_modular_old_rt copy another_5am_conc.py
   ```

## Project Structure

- `reconstruction_main_both_HF_VHF_withdelT_modular_old_rt copy another_5am_conc.py`: Main tomography script
- Supporting modules: `modify_df.py`, `ionosphere_design.py`, `ray_trace_passive.py`, `improved_geometry.py`
- `data/`: Input data files
- `img/`: Output figures

## Citation

If you use this code, please cite the associated AGU abstract and references above.
