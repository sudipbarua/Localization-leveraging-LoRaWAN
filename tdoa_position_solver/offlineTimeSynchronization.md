
**Offline Time Synchronization (MLE-LP)**

- **File:** `offline_time_sync_MLE-LP.py`

**Project Description:**
- **Summary:**: Implementation of an offline time-synchronization routine that estimates end-device and gateway clock offsets from recorded TDoA/Timestamp observations. The algorithm follows the MLE-LP approach described in the referenced IEEE paper and is intended for post-processing of collected LoRaWAN measurements.
- **Context:**: This routine is part of the `tdoa_position_solver` tools in this repository and is useful when timestamps from multiple gateways must be aligned after data collection in order to support range-based or TDoA localization.

**Objectives:**
- **Estimate clock offsets:**: Compute per-device and/or per-gateway time offsets using a Maximum Likelihood Estimation formulation.
- **Robust offline processing:**: Provide a stable routine that works on stored measurement files (CSV/TSV) without needing online synchronization.
- **Provide usable outputs:**: Save corrected timestamps and offset estimates suitable for downstream position estimation algorithms in this repo.

**Method Overview (brief):**
- **Approach:**: The implementation is based on the MLE-LP method from the referenced paper. The problem is cast as a maximum-likelihood estimation of clock offsets given noisy timestamp differences; an optimization (LP relaxation and iterative refinement) is used to find the offsets that best explain the observed measurements.
- **Assumptions:**: The measurements include enough cross-references (same transmissions seen by multiple gateways) so that relative offsets can be estimated. Measurement noise is assumed approximately Gaussian; the MLE formulation accounts for variance in timing measurements.

**Inputs:**
- **Primary measurement file:**: A CSV (or other delimited) file containing recorded packet reception timestamps and identifiers. Typical columns the script expects (or will accept) are:
	- `device_id` or `dev_eui`: End-device identifier
	- `gw_id`: Gateway identifier
	- `rx_timestamp`: Gateway reception timestamp (seconds or microseconds; be consistent)
	- `tx_timestamp` (optional): Transmit timestamp from device, if available
	- `rssi` / `snr` (optional): useful for diagnostics but not required by the estimator
- **Gateway positions (optional):**: A CSV with `gw_id,x,y,z` if position information is needed by downstream steps. The offline time-sync itself primarily uses timestamps, but many workflows combine the offsets with GW positions for localization.
- **Configuration / parameters:**: Noise variances, solver options, and convergence thresholds may be configured either via command-line flags or by editing the script constants. Check the top of `offline_time_sync_MLE-LP.py` for parameters.

**Outputs:**
- **Offsets file:**: CSV containing estimated clock offsets per gateway or per device, e.g., `gw_id,estimated_offset,offset_std`.
- **Corrected timestamps (optional):**: Measurement file with an extra column `corrected_rx_timestamp` producing timestamps aligned to the chosen reference clock.
- **Diagnostics / logs:**: Convergence information, final loss value, and any solver warnings printed to stdout or saved to a log file (depending on script settings).

**Usage:**
- **Prerequisites:**: Python 3.8+ and dependencies listed in `requirements.txt` for the repository. Install with:

```
pip install -r requirements.txt
```

- **Basic run:**: From the repository root you can run the script. Example (replace arguments as needed):

```
python tdoa_position_solver/offline_time_sync_MLE-LP.py \
	--input data/measurements.csv \
	--gw-positions data/gw_positions.csv \
	--output results/offset_estimates.csv
```

- **Notes on arguments:**: The script's exact CLI switches may vary; inspect the script header or run `--help` for the canonical list. The typical arguments are:
	- `--input` : path to measurement CSV
	- `--gw-positions` : (optional) path to gateway positions CSV
	- `--output` : output path for offsets / corrected timestamps
	- `--reference` : choose which clock is the reference (e.g., a gateway id or `median`)
	- `--verbose` / `--log` : enable extra debugging output

**Example data layout (CSV):**
- `measurements.csv` (comma-separated)

```
device_id,gw_id,rx_timestamp,tx_timestamp,rssi
dev-001,gw-01,1580000000.123456,1580000000.000000,-75
dev-001,gw-02,1580000000.124012,1580000000.000000,-80
dev-002,gw-01,1580000001.523201,1580000001.400000,-70
```

**Reference:**
- The implementation and algorithm are based on the following paper: 
- **Source:**: https://ieeexplore.ieee.org/abstract/document/4814517

**Practical Notes and Tips:**
- **Units:**: Ensure timestamps use consistent units (seconds or microseconds). Mixing units will break the estimator.
- **Reference clock:**: The algorithm computes relative offsets. Select a stable reference clock (a gateway with accurate time) or use the median offset as reference.
- **Data sufficiency:**: The estimator requires that many packets are observed by multiple gateways; single-observation devices/gateways cannot be reliably synchronized alone.
- **Inspect script:**: `offline_time_sync_MLE-LP.py` includes implementation details and may provide more CLI options — open it to tailor inputs and solver settings.

If you want, I can:
- add an example measurement file with synthetic data, or
- update `offline_time_sync_MLE-LP.py` to include a `--help`-driven CLI with clearer argument parsing, or
- run a quick test on a sample dataset in this repo and show results.

