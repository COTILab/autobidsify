# nirs_convert.py - COMPLETE v10
# Full implementation with .mat and .nirs direct conversion

"""
fNIRS Converter Module - Complete v10

Supported conversions:
1. MATLAB .mat → SNIRF (NEW: direct conversion)
2. Homer3 .nirs → SNIRF (NEW: direct conversion)
3. CSV/TSV tables → SNIRF (existing: using normalized headers)
4. SNIRF sidecar generation
5. SNIRF validation

Dependencies:
- h5py: SNIRF file creation
- numpy: Data manipulation
- scipy: .mat file loading
- csv: CSV parsing (built-in)
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import h5py
import csv
from autobidsify.utils import ensure_dir, warn, info, write_json


# ============================================================================
# NEW v10: Direct MATLAB .mat to SNIRF conversion
# ============================================================================

def convert_mat_to_snirf(mat_file: Path, output_path: Path, 
                        quiet: bool = False) -> Optional[Path]:
    """
    Convert MATLAB .mat file to SNIRF format.
    
    Strategy:
    1. Load .mat file using scipy.io.loadmat
    2. Detect fNIRS data structure (common variable names: d, data, t, time)
    3. Extract time series and metadata
    4. Create SNIRF HDF5 file with required groups
    
    Args:
        mat_file: Path to .mat file
        output_path: Output SNIRF path (.snirf extension)
        quiet: Suppress output messages
    
    Returns:
        Path to created SNIRF file, or None if conversion failed
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        if not quiet:
            warn("scipy not installed")
            warn("Install with: pip install scipy")
        return None
    
    if not mat_file.exists():
        if not quiet:
            warn(f"MAT file not found: {mat_file}")
        return None
    
    try:
        if not quiet:
            info(f"  Converting MATLAB: {mat_file.name}")
        
        # Load .mat file
        mat_data = loadmat(str(mat_file))
        
        # Common fNIRS variable names to check
        data_var_names = ['d', 'data', 'dOD', 'dConc', 'y', 'timeseries', 'nirs_data']
        time_var_names = ['t', 'time', 'times', 'time_vector']
        
        # Find data variable
        data_array = None
        data_var = None
        for var_name in data_var_names:
            if var_name in mat_data:
                data_array = mat_data[var_name]
                data_var = var_name
                break
        
        if data_array is None:
            # List available variables
            available = [k for k in mat_data.keys() if not k.startswith('__')]
            if not quiet:
                warn(f"  Could not find fNIRS data variable")
                warn(f"  Available variables: {available}")
                warn(f"  Expected one of: {data_var_names}")
            return None
        
        # Find time variable
        time_array = None
        for var_name in time_var_names:
            if var_name in mat_data:
                time_array = mat_data[var_name]
                break
        
        # Generate time vector if not found
        if time_array is None:
            n_samples = data_array.shape[0]
            time_array = np.arange(n_samples) / 10.0  # Assume 10 Hz
            if not quiet:
                warn(f"  No time vector found, generated assuming 10 Hz sampling")
        
        # Ensure correct shape
        if len(data_array.shape) == 1:
            data_array = data_array.reshape(-1, 1)
        
        if len(time_array.shape) > 1:
            time_array = time_array.flatten()
        
        n_samples, n_channels = data_array.shape
        
        if not quiet:
            info(f"  Data shape: {data_array.shape} (samples × channels)")
            info(f"  Time samples: {len(time_array)}")
        
        # Check dimension consistency
        if n_samples != len(time_array):
            if not quiet:
                warn(f"  Dimension mismatch: data {n_samples}, time {len(time_array)}")
            # Try to fix by truncating to minimum length
            min_len = min(n_samples, len(time_array))
            data_array = data_array[:min_len, :]
            time_array = time_array[:min_len]
            n_samples = min_len
            if not quiet:
                info(f"  → Adjusted to {n_samples} samples")
        
        # Extract wavelengths if available
        wavelengths = [760, 850]  # Default NIR wavelengths
        if 'SD' in mat_data:
            # Homer3 style SD structure
            try:
                SD = mat_data['SD']
                if isinstance(SD, np.ndarray) and SD.dtype.names:
                    if 'Lambda' in SD.dtype.names:
                        wl = SD['Lambda'][0, 0].flatten()
                        if len(wl) > 0:
                            wavelengths = wl.tolist()
                            if not quiet:
                                info(f"  Extracted wavelengths: {wavelengths}")
            except:
                pass
        
        # Create SNIRF file
        ensure_dir(output_path.parent)
        
        with h5py.File(output_path, 'w') as f:
            # /nirs group (required)
            nirs_grp = f.create_group("nirs")
            
            # /nirs/data1 (required)
            data_grp = nirs_grp.create_group("data1")
            data_grp.create_dataset("dataTimeSeries", data=data_array, dtype='f')
            data_grp.create_dataset("time", data=time_array, dtype='f')
            
            # /nirs/data1/measurementList (required)
            ml_grp = data_grp.create_group("measurementList")
            
            for ch_idx in range(n_channels):
                ch_grp = ml_grp.create_group(str(ch_idx + 1))
                
                # Simplified channel mapping (1 source, multiple detectors)
                ch_grp.create_dataset("sourceIndex", data=1)
                ch_grp.create_dataset("detectorIndex", data=ch_idx + 1)
                ch_grp.create_dataset("wavelengthIndex", data=1)
                ch_grp.create_dataset("dataType", data=1)  # 1 = Intensity
                ch_grp.create_dataset("dataTypeLabel", data="Intensity")
            
            # /nirs/probe (required)
            probe_grp = nirs_grp.create_group("probe")
            probe_grp.create_dataset("wavelengths", data=wavelengths)
            
            # Simplified probe geometry
            n_sources = 1
            n_detectors = n_channels
            probe_grp.create_dataset("sourcePos2D", data=np.zeros((n_sources, 2)))
            probe_grp.create_dataset("detectorPos2D", data=np.zeros((n_detectors, 2)))
            
            # /nirs/metaDataTags (required)
            meta_grp = nirs_grp.create_group("metaDataTags")
            meta_grp.create_dataset("SubjectID", data="unknown")
            meta_grp.create_dataset("MeasurementDate", data="")
            meta_grp.create_dataset("MeasurementTime", data="")
            meta_grp.create_dataset("LengthUnit", data="mm")
            meta_grp.create_dataset("TimeUnit", data="s")
            meta_grp.create_dataset("FrequencyUnit", data="Hz")
        
        if not quiet:
            info(f"  ✓ Created SNIRF: {output_path.name}")
            info(f"    Channels: {n_channels}, Samples: {n_samples}")
        
        # Validate created file
        if validate_snirf_file(output_path):
            return output_path
        else:
            if not quiet:
                warn(f"  Created file failed validation")
            return None
        
    except Exception as e:
        if not quiet:
            warn(f"  MAT→SNIRF conversion failed: {e}")
            import traceback
            traceback.print_exc()
        return None


# ============================================================================
# NEW v10: Homer3 .nirs to SNIRF conversion
# ============================================================================

def convert_nirs_to_snirf(nirs_file: Path, output_path: Path,
                         quiet: bool = False) -> Optional[Path]:
    """
    Convert Homer3 .nirs file to SNIRF format.
    
    Homer3 .nirs file structure (it's actually a MATLAB .mat file):
    - d: data matrix (samples × channels)
    - t: time vector (samples)
    - SD: source-detector configuration (optional)
    - s: stimulus markers (optional)
    
    Args:
        nirs_file: Path to .nirs file
        output_path: Output SNIRF path (.snirf extension)
        quiet: Suppress output messages
    
    Returns:
        Path to created SNIRF file, or None if conversion failed
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        if not quiet:
            warn("scipy not installed for .nirs conversion")
            warn("Install with: pip install scipy")
        return None
    
    if not nirs_file.exists():
        if not quiet:
            warn(f".nirs file not found: {nirs_file}")
        return None
    
    try:
        if not quiet:
            info(f"  Converting Homer3 .nirs: {nirs_file.name}")
        
        # Load .nirs file (it's a MATLAB .mat file with specific structure)
        nirs_data = loadmat(str(nirs_file))
        
        # Extract data matrix (d variable is standard in Homer3)
        if 'd' not in nirs_data:
            if not quiet:
                warn(f"  'd' variable not found in {nirs_file.name}")
                available = [k for k in nirs_data.keys() if not k.startswith('__')]
                warn(f"  Available: {available}")
            return None
        
        data_array = nirs_data['d']
        
        # Extract time vector (t variable)
        if 't' in nirs_data:
            time_array = nirs_data['t'].flatten()
        else:
            # Generate time vector if not found
            n_samples = data_array.shape[0]
            time_array = np.arange(n_samples) / 10.0  # Assume 10 Hz
            if not quiet:
                warn(f"  No time vector ('t'), generated assuming 10 Hz")
        
        # Ensure correct dimensions
        if len(data_array.shape) == 1:
            data_array = data_array.reshape(-1, 1)
        
        if len(time_array.shape) > 1:
            time_array = time_array.flatten()
        
        n_samples, n_channels = data_array.shape
        
        if not quiet:
            info(f"  Data shape: {data_array.shape}")
            info(f"  Channels: {n_channels}, Samples: {n_samples}")
        
        # Check dimension consistency
        if n_samples != len(time_array):
            if not quiet:
                warn(f"  Dimension mismatch: data {n_samples}, time {len(time_array)}")
            min_len = min(n_samples, len(time_array))
            data_array = data_array[:min_len, :]
            time_array = time_array[:min_len]
            n_samples = min_len
            if not quiet:
                info(f"  → Adjusted to {n_samples} samples")
        
        # Extract wavelengths from SD structure (Homer3 specific)
        wavelengths = [760, 850]  # Default
        n_sources = 1
        n_detectors = n_channels
        
        if 'SD' in nirs_data:
            try:
                SD = nirs_data['SD']
                
                # SD is a structured array in Homer3
                if isinstance(SD, np.ndarray) and SD.dtype.names:
                    # Extract wavelengths
                    if 'Lambda' in SD.dtype.names:
                        wl = SD['Lambda'][0, 0].flatten()
                        if len(wl) > 0:
                            wavelengths = wl.tolist()
                    
                    # Extract source/detector positions if available
                    if 'SrcPos' in SD.dtype.names:
                        src_pos = SD['SrcPos'][0, 0]
                        if len(src_pos) > 0:
                            n_sources = src_pos.shape[0]
                    
                    if 'DetPos' in SD.dtype.names:
                        det_pos = SD['DetPos'][0, 0]
                        if len(det_pos) > 0:
                            n_detectors = det_pos.shape[0]
                
                if not quiet:
                    info(f"  Wavelengths: {wavelengths} nm")
                    info(f"  Sources: {n_sources}, Detectors: {n_detectors}")
            except Exception as e:
                if not quiet:
                    warn(f"  Could not parse SD structure: {e}")
        
        # Create SNIRF file
        ensure_dir(output_path.parent)
        
        with h5py.File(output_path, 'w') as f:
            # /nirs group
            nirs_grp = f.create_group("nirs")
            
            # /nirs/data1
            data_grp = nirs_grp.create_group("data1")
            data_grp.create_dataset("dataTimeSeries", data=data_array, dtype='f')
            data_grp.create_dataset("time", data=time_array, dtype='f')
            
            # /nirs/data1/measurementList
            ml_grp = data_grp.create_group("measurementList")
            
            for ch_idx in range(n_channels):
                ch_grp = ml_grp.create_group(str(ch_idx + 1))
                
                # Simplified: assume sequential source/detector mapping
                # In production, this should be extracted from SD.MeasList
                ch_grp.create_dataset("sourceIndex", data=1)
                ch_grp.create_dataset("detectorIndex", data=ch_idx + 1)
                ch_grp.create_dataset("wavelengthIndex", data=1)
                ch_grp.create_dataset("dataType", data=1)  # 1 = Intensity
                ch_grp.create_dataset("dataTypeLabel", data="Intensity")
            
            # /nirs/probe
            probe_grp = nirs_grp.create_group("probe")
            probe_grp.create_dataset("wavelengths", data=wavelengths)
            
            # Simplified probe geometry (2D positions)
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
        
        if not quiet:
            info(f"  ✓ Created SNIRF: {output_path.name}")
            info(f"    Data: {n_samples} samples × {n_channels} channels")
        
        # Validate
        if validate_snirf_file(output_path):
            return output_path
        else:
            return None
        
    except Exception as e:
        if not quiet:
            warn(f"  MAT→SNIRF conversion failed: {e}")
            import traceback
            traceback.print_exc()
        return None


# ============================================================================
# NEW v10: Homer3 .nirs to SNIRF conversion
# ============================================================================

def convert_nirs_to_snirf(nirs_file: Path, output_path: Path,
                         quiet: bool = False) -> Optional[Path]:
    """
    Convert Homer3 .nirs file to SNIRF format.
    
    Note: .nirs files ARE .mat files with Homer3-specific structure.
    This function is essentially the same as convert_mat_to_snirf() but
    with Homer3-specific variable name expectations.
    
    Args:
        nirs_file: Path to .nirs file (Homer3 format)
        output_path: Output SNIRF path (.snirf extension)
        quiet: Suppress output messages
    
    Returns:
        Path to created SNIRF file, or None if conversion failed
    """
    # .nirs files are .mat files, so we can use the same converter
    return convert_mat_to_snirf(nirs_file, output_path, quiet=quiet)


# ============================================================================
# Existing: CSV/TSV to SNIRF (uses normalized headers from LLM)
# ============================================================================

def write_snirf_from_normalized(
    normalized: Dict[str, Any],
    input_root: Path,
    output_dir: Path
) -> List[Path]:
    """
    Create SNIRF files from CSV/table data using LLM-generated normalized headers.
    
    This function is used when:
    - Input data is in CSV/TSV format
    - LLM has generated normalized column mappings
    - Complex header structures need semantic understanding
    
    For direct .mat/.nirs conversion, use:
    - convert_mat_to_snirf() for .mat files
    - convert_nirs_to_snirf() for .nirs files
    
    Args:
        normalized: Normalized headers from LLM (contains column mappings)
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
    
    info(f"CSV→SNIRF conversion parameters:")
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
        
        info(f"Processing CSV: {relpath}")
        
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
                # /nirs group
                nirs_grp = f.create_group("nirs")
                
                # /nirs/data1
                data_grp = nirs_grp.create_group("data1")
                data_grp.create_dataset("dataTimeSeries", data=data_matrix, dtype='f')
                data_grp.create_dataset("time", data=time_data, dtype='f')
                
                # /nirs/data1/measurementList
                ml_grp = data_grp.create_group("measurementList")
                
                for ch_idx in range(n_channels):
                    ch_grp = ml_grp.create_group(str(ch_idx + 1))
                    
                    # Simplified: sequential source/detector
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
            
            info(f"  ✓ Created SNIRF: {snirf_path.name}")
            snirf_files.append(snirf_path)
            
            # Validate
            validate_snirf_file(snirf_path)
            
        except Exception as e:
            warn(f"Failed to process {relpath}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return snirf_files


# ============================================================================
# Sidecar generation
# ============================================================================

def write_nirs_sidecars(output_dir: Path, defaults: Dict[str, Any]) -> None:
    """
    Create BIDS sidecar files for all SNIRF files in directory.
    
    Creates for each .snirf file:
        - *_nirs.json: Task metadata (required)
        - *_channels.tsv: Channel descriptions (recommended)
        - *_optodes.tsv: Optode positions (recommended)
    
    Args:
        output_dir: Directory containing SNIRF files
        defaults: Default metadata values (TaskName, SamplingFrequency, etc.)
    """
    snirf_files = list(Path(output_dir).rglob("*.snirf"))
    
    if not snirf_files:
        return
    
    info(f"Generating sidecars for {len(snirf_files)} SNIRF files...")
    
    for snirf_file in snirf_files:
        stem = snirf_file.stem
        
        # Remove _nirs suffix if already present (avoid duplication)
        if stem.endswith('_nirs'):
            stem = stem[:-5]
        
        # Create *_nirs.json (required BIDS sidecar)
        json_path = snirf_file.parent / f"{stem}_nirs.json"
        if not json_path.exists():
            sidecar = {
                "TaskName": defaults.get("TaskName", "rest"),
                "SamplingFrequency": defaults.get("SamplingFrequency", 10.0),
                "NIRSChannelCount": defaults.get("NIRSChannelCount", 0),
                "NIRSSourceOptodeCount": defaults.get("NIRSSourceOptodeCount", 1),
                "NIRSDetectorOptodeCount": defaults.get("NIRSDetectorOptodeCount", 1)
            }
            write_json(json_path, sidecar)
            info(f"  ✓ {json_path.name}")
        
        # Create *_channels.tsv (recommended)
        channels_path = snirf_file.parent / f"{stem}_channels.tsv"
        if not channels_path.exists():
            with open(channels_path, 'w') as f:
                f.write("name\ttype\tsource\tdetector\twavelength_nominal\tunits\n")
                # Placeholder - should be extracted from SNIRF file
                f.write("ch1\tNIRS\t1\t1\t760\tnm\n")
            info(f"  ✓ {channels_path.name}")
        
        # Create *_optodes.tsv (recommended)
        optodes_path = snirf_file.parent / f"{stem}_optodes.tsv"
        if not optodes_path.exists():
            with open(optodes_path, 'w') as f:
                f.write("name\ttype\tx\ty\tz\n")
                # Placeholder positions
                f.write("S1\tsource\t0\t0\t0\n")
                f.write("D1\tdetector\t30\t0\t0\n")
            info(f"  ✓ {optodes_path.name}")


# ============================================================================
# SNIRF validation
# ============================================================================

def validate_snirf_file(snirf_path: Path) -> bool:
    """
    Validate SNIRF file structure according to SNIRF specification.
    
    Checks:
        - Valid HDF5 file format
        - Required groups exist: /nirs, /nirs/data1
        - Required datasets: dataTimeSeries, time
        - Dimension consistency: data and time have matching sample counts
        - MeasurementList structure
    
    Args:
        snirf_path: Path to SNIRF file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        with h5py.File(snirf_path, 'r') as f:
            # Check /nirs group (required)
            if "nirs" not in f:
                warn(f"SNIRF invalid: missing /nirs group")
                return False
            
            nirs = f["nirs"]
            
            # Check /nirs/data1 (required)
            if "data1" not in nirs:
                warn(f"SNIRF invalid: missing /nirs/data1")
                return False
            
            data1 = nirs["data1"]
            
            # Check required datasets
            if "dataTimeSeries" not in data1:
                warn(f"SNIRF invalid: missing dataTimeSeries")
                return False
            
            if "time" not in data1:
                warn(f"SNIRF invalid: missing time vector")
                return False
            
            # Check dimension consistency
            data_shape = data1["dataTimeSeries"].shape
            time_shape = data1["time"].shape
            
            if data_shape[0] != time_shape[0]:
                warn(f"SNIRF invalid: dimension mismatch")
                warn(f"  dataTimeSeries: {data_shape}, time: {time_shape}")
                return False
            
            # Check measurementList (required)
            if "measurementList" not in data1:
                warn(f"SNIRF invalid: missing measurementList")
                return False
            
            info(f"  ✓ SNIRF valid: {snirf_path.name}")
            info(f"    Shape: {data_shape[0]} samples × {data_shape[1]} channels")
            return True
            
    except Exception as e:
        warn(f"SNIRF validation error: {e}")
        return False


# ============================================================================
# LEGACY: MATLAB toolbox conversion (placeholder)
# ============================================================================

def run_homer3_nirs_to_snirf(nirs_files: List[Path], output_dir: Path) -> List[Path]:
    """
    Convert Homer3 .nirs files using Homer3 MATLAB toolbox.
    
    LEGACY function - kept for compatibility.
    
    NOTE: This requires MATLAB + Homer3 installation.
    For most users, use the pure Python convert_nirs_to_snirf() instead.
    
    Args:
        nirs_files: List of .nirs file paths
        output_dir: Output directory
    
    Returns:
        List of generated SNIRF files
    """
    warn("Homer3 MATLAB toolbox conversion not implemented")
    info("Using pure Python converter instead...")
    
    # Call pure Python converter for each file
    converted = []
    for nirs_file in nirs_files:
        output_path = output_dir / (nirs_file.stem + ".snirf")
        result = convert_nirs_to_snirf(nirs_file, output_path, quiet=False)
        if result:
            converted.append(result)
    
    return converted