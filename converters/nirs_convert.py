# nirs_convert.py
# NIRS format converters: CSV/tables→SNIRF, Homer3 .nirs→SNIRF

"""
The NIRS converter module handles fNIRS data format conversion and SNIRF file generation.

Supported conversions:
1. Homer3 .nirs → SNIRF (using MATLAB tools, optional)
2. CSV/TSV tables → SNIRF (custom parser, core functionality)
3. SNIRF verification and BIDS-side file generation

CSV to SNIRF conversion process (write_snirf_from_normalized):

1. Read configuration from normalized_headers
2. Read specified columns from the CSV file
3. Construct a dataTimeSeries matrix (samples × channels)
4. Create HDF5 file structure:
/nirs/data1/dataTimeSeries
/nirs/data1/time
/nirs/data1/measurementList/
/nirs/probe/wavelengths
/nirs/metaDataTags/
5. Verify SNIRF structure integrity
Key technologies:
- h5py for creating HDF5 files
- numpy for constructing data matrices
- CSV parsing (supports multiple delimiters)
- Time unit is uniformly set to seconds
- Channel index starts from 1 (SNIRF specification)

"""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import h5py
import csv
from utils import ensure_dir, warn, info, write_json

def run_homer3_nirs_to_snirf(nirs_files: List[Path], output_dir: Path) -> List[Path]:
    """
    Convert Homer3 .nirs files to SNIRF using Homer3 toolbox.
    
    NOTE: This requires MATLAB + Homer3 to be installed.
    This is a placeholder implementation.
    
    Args:
        nirs_files: List of .nirs file paths
        output_dir: Output directory
    
    Returns:
        List of generated SNIRF files
    """
    warn("Homer3 conversion requires MATLAB + Homer3 toolbox")
    warn("This is a placeholder. Skipping .nirs files.")
    warn("For production use, implement MATLAB bridge or use standalone Homer3 CLI")
    
    return []

def write_snirf_from_normalized(
    normalized: Dict[str, Any],
    input_root: Path,
    output_dir: Path
) -> List[Path]:
    """
    Create SNIRF files from CSV/table data using normalized headers.
    
    This is the core custom parser that converts tabular fNIRS data
    to SNIRF format based on LLM-generated normalized configuration.
    
    Args:
        normalized: Normalized headers from LLM
        input_root: Root directory of input data
        output_dir: Output directory for SNIRF files
    
    Returns:
        List of created SNIRF file paths
    """
    ensure_dir(output_dir)
    
    snirf_files = []
    
    # Extract global settings
    norm_data = normalized.get("normalized", {})
    globals_dict = norm_data.get("globals", {})
    
    sampling_freq = globals_dict.get("SamplingFrequency", 10.0)
    wavelengths = globals_dict.get("Wavelengths", [760, 850])
    task_name = globals_dict.get("TaskName", "task")
    
    info(f"SNIRF conversion parameters:")
    info(f"  SamplingFrequency: {sampling_freq} Hz")
    info(f"  Wavelengths: {wavelengths} nm")
    info(f"  TaskName: {task_name}")
    
    # Process each file
    for file_spec in norm_data.get("files", []):
        relpath = file_spec.get("relpath", "")
        input_path = input_root / relpath
        
        if not input_path.exists():
            warn(f"Input file not found: {input_path}")
            continue
        
        info(f"Processing: {relpath}")
        
        try:
            # Read CSV data
            time_spec = file_spec.get("time", {})
            time_col = time_spec.get("column", "time")
            time_unit = time_spec.get("unit", "s")
            
            signals = file_spec.get("signals", [])
            
            # Parse CSV
            data_dict = {}
            with open(input_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                # Strip whitespace from headers
                if reader.fieldnames:
                    reader.fieldnames = [name.strip() for name in reader.fieldnames]
                
                for row in reader:
                    for key, value in row.items():
                        if key not in data_dict:
                            data_dict[key] = []
                        try:
                            data_dict[key].append(float(value))
                        except (ValueError, TypeError):
                            data_dict[key].append(0.0)
            
            # Extract time vector
            if time_col not in data_dict:
                warn(f"Time column '{time_col}' not found in {relpath}")
                continue
            
            time_data = np.array(data_dict[time_col])
            
            # Convert time unit to seconds
            if time_unit == "milliseconds":
                time_data = time_data / 1000.0
            elif time_unit == "minutes":
                time_data = time_data * 60.0
            
            n_samples = len(time_data)
            info(f"  Time samples: {n_samples}")
            
            # Build dataTimeSeries matrix
            all_channels = []
            channel_info = []
            
            for signal in signals:
                signal_type = signal.get("type", "Intensity")
                columns = signal.get("columns", [])
                
                for col in columns:
                    col_stripped = col.strip()
                    if col_stripped in data_dict:
                        all_channels.append(data_dict[col_stripped])
                        channel_info.append({
                            "type": signal_type,
                            "column": col_stripped
                        })
            
            if not all_channels:
                warn(f"No valid channels found in {relpath}")
                continue
            
            data_matrix = np.array(all_channels).T  # (n_samples, n_channels)
            n_channels = data_matrix.shape[1]
            
            info(f"  Data matrix: {data_matrix.shape} (samples × channels)")
            
            # Create SNIRF file
            output_name = input_path.stem + ".snirf"
            snirf_path = output_dir / output_name
            
            with h5py.File(snirf_path, 'w') as f:
                # Create /nirs group
                nirs_grp = f.create_group("nirs")
                
                # /nirs/data1
                data_grp = nirs_grp.create_group("data1")
                data_grp.create_dataset("dataTimeSeries", data=data_matrix, dtype='f')
                data_grp.create_dataset("time", data=time_data, dtype='f')
                
                # /nirs/data1/measurementList
                ml_grp = data_grp.create_group("measurementList")
                
                for ch_idx in range(n_channels):
                    ch_grp = ml_grp.create_group(str(ch_idx + 1))
                    
                    # Simplified: assume sequential source/detector
                    ch_grp.create_dataset("sourceIndex", data=1)
                    ch_grp.create_dataset("detectorIndex", data=ch_idx + 1)
                    ch_grp.create_dataset("wavelengthIndex", data=1)
                    
                    # dataType: 1=Intensity, 99=processed
                    data_type = 1 if channel_info[ch_idx]["type"] == "Intensity" else 99
                    ch_grp.create_dataset("dataType", data=data_type)
                    ch_grp.create_dataset("dataTypeLabel", data=channel_info[ch_idx]["type"])
                
                # /nirs/probe
                probe_grp = nirs_grp.create_group("probe")
                probe_grp.create_dataset("wavelengths", data=wavelengths)
                
                # Simplified probe geometry
                n_sources = 1
                n_detectors = n_channels
                probe_grp.create_dataset("sourcePos2D", data=np.zeros((n_sources, 2)))
                probe_grp.create_dataset("detectorPos2D", data=np.zeros((n_detectors, 2)))
                
                # /nirs/metaDataTags
                meta_grp = nirs_grp.create_group("metaDataTags")
                meta_grp.create_dataset("SubjectID", data="unknown")
                meta_grp.create_dataset("MeasurementDate", data="")
                meta_grp.create_dataset("MeasurementTime", data="")
                meta_grp.create_dataset("LengthUnit", data="mm")
                meta_grp.create_dataset("TimeUnit", data="s")
                meta_grp.create_dataset("FrequencyUnit", data="Hz")
            
            info(f"  ✓ Created SNIRF: {snirf_path}")
            snirf_files.append(snirf_path)
            
            # Validate
            validate_snirf_file(snirf_path)
            
        except Exception as e:
            warn(f"Failed to process {relpath}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return snirf_files

def write_nirs_sidecars(output_dir: Path, defaults: Dict[str, Any]) -> None:
    """
    Create BIDS sidecar files for all SNIRF files in directory.
    
    Creates:
        - *_nirs.json: Task metadata
        - *_channels.tsv: Channel descriptions
        - *_optodes.tsv: Optode positions
    
    Args:
        output_dir: Directory containing SNIRF files
        defaults: Default metadata values
    """
    for snirf_file in Path(output_dir).rglob("*.snirf"):
        stem = snirf_file.stem
        
        # Create _nirs.json
        json_path = snirf_file.with_suffix('.json')
        if not json_path.exists():
            sidecar = {
                "TaskName": defaults.get("TaskName", "task"),
                "SamplingFrequency": defaults.get("SamplingFrequency", 10.0),
                "NIRSChannelCount": 0,
                "NIRSSourceOptodeCount": 1,
                "NIRSDetectorOptodeCount": 1
            }
            write_json(json_path, sidecar)
            info(f"  ✓ Created sidecar: {json_path.name}")
        
        # Create _channels.tsv
        channels_path = snirf_file.parent / f"{stem}_channels.tsv"
        if not channels_path.exists():
            with open(channels_path, 'w') as f:
                f.write("name\ttype\tsource\tdetector\twavelength_nominal\tunits\n")
                f.write("ch1\tNIRS\t1\t1\t760\tnm\n")
            info(f"  ✓ Created: {channels_path.name}")
        
        # Create _optodes.tsv
        optodes_path = snirf_file.parent / f"{stem}_optodes.tsv"
        if not optodes_path.exists():
            with open(optodes_path, 'w') as f:
                f.write("name\ttype\tx\ty\tz\n")
                f.write("S1\tsource\t0\t0\t0\n")
                f.write("D1\tdetector\t30\t0\t0\n")
            info(f"  ✓ Created: {optodes_path.name}")

def validate_snirf_file(snirf_path: Path) -> bool:
    """
    Validate SNIRF file structure.
    
    Checks:
        - Valid HDF5 format
        - Required groups exist
        - Data consistency
    
    Args:
        snirf_path: Path to SNIRF file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        with h5py.File(snirf_path, 'r') as f:
            # Check /nirs group
            if "nirs" not in f:
                warn(f"SNIRF invalid: missing /nirs group in {snirf_path}")
                return False
            
            nirs = f["nirs"]
            
            # Check /nirs/data1
            if "data1" not in nirs:
                warn(f"SNIRF invalid: missing /nirs/data1 in {snirf_path}")
                return False
            
            data1 = nirs["data1"]
            
            # Check required datasets
            if "dataTimeSeries" not in data1:
                warn(f"SNIRF invalid: missing dataTimeSeries in {snirf_path}")
                return False
            
            if "time" not in data1:
                warn(f"SNIRF invalid: missing time in {snirf_path}")
                return False
            
            # Check dimension consistency
            data_shape = data1["dataTimeSeries"].shape
            time_shape = data1["time"].shape
            
            if data_shape[0] != time_shape[0]:
                warn(f"SNIRF invalid: dimension mismatch in {snirf_path}")
                warn(f"  dataTimeSeries: {data_shape}, time: {time_shape}")
                return False
            
            info(f"  ✓ SNIRF valid: {snirf_path.name}")
            return True
            
    except Exception as e:
        warn(f"SNIRF validation error for {snirf_path}: {e}")
        return False
