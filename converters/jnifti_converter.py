# converters/jnifti_convert.py
# Pure Python implementation of JNIfTI → NIfTI converter
# Translated from MATLAB jnii2nii.m (NeuroJSON/jnifty)

"""
JNIfTI to NIfTI Converter - Pure Python Implementation

Converts JNIfTI files (.jnii for text JSON, .bnii for binary JSON) to NIfTI format.

JNIfTI Format:
- Text-based (.jnii): JSON format with NIFTIHeader + NIFTIData
- Binary (.bnii): Binary JSON (BJData/UBJSON) format

Dependencies:
- nibabel: NIfTI file I/O
- numpy: Array operations
- json: JSON parsing (built-in)
- bjdata: Binary JSON parsing (for .bnii files)

Installation:
    pip install nibabel numpy bjdata

Original MATLAB implementation: https://github.com/NeuroJSON/jnifty/blob/master/jnii2nii.m
License: Apache 2.0
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import numpy as np
import nibabel as nib
from utils import ensure_dir, warn, info

# ============================================================================
# NIfTI Code Maps (from niicodemap.m)
# ============================================================================

NIFTI_INTENT_CODES = {
    'none': 0, 'cor': 2, 'ttest': 3, 'ftest': 4, 'zscore': 5,
    'chisq': 6, 'beta': 7, 'binom': 8, 'gamma': 9, 'poisson': 10,
    'normal': 11, 'ftest_nonc': 12, 'chisq_nonc': 13, 'logistic': 14,
    'laplace': 15, 'uniform': 16, 'ttest_nonc': 17, 'weibull': 18,
    'chi': 19, 'invgauss': 20, 'extval': 21, 'pval': 22, 'logpval': 23,
    'log10pval': 24, 'estimate': 1001, 'label': 1002, 'neuroname': 1003,
    'genmatrix': 1004, 'symmatrix': 1005, 'dispvect': 1006, 'vector': 1007,
    'pointset': 1008, 'triangle': 1009, 'quaternion': 1010, 'dimless': 1011,
    'time_series': 2001, 'node_index': 2002, 'rgb_vector': 2003,
    'rgba_vector': 2004, 'shape': 2005
}

NIFTI_DATATYPE_CODES = {
    'uint8': 2, 'int16': 4, 'int32': 8, 'float32': 16, 'complex64': 32,
    'float64': 64, 'rgb24': 128, 'int8': 256, 'uint16': 512, 'uint32': 768,
    'int64': 1024, 'uint64': 1280, 'float128': 1536, 'complex128': 1792,
    'complex256': 2048, 'rgba32': 2304
}

NIFTI_SLICE_CODES = {
    'unknown': 0, 'seq_inc': 1, 'seq_dec': 2, 'alt_inc': 3,
    'alt_dec': 4, 'alt_inc2': 5, 'alt_dec2': 6
}

NIFTI_XFORM_CODES = {
    'unknown': 0, 'scanner': 1, 'aligned': 2, 'talairach': 3, 'mni': 4
}

NIFTI_UNIT_CODES = {
    'unknown': 0, 'meter': 1, 'mm': 2, 'micron': 3,
    'sec': 8, 'msec': 16, 'usec': 24, 'hz': 32, 'ppm': 40, 'rads': 48
}


# ============================================================================
# Core Conversion Functions
# ============================================================================

def check_jnifti_support() -> bool:
    """
    Check if JNIfTI conversion is supported.
    
    Returns:
        True if all required dependencies are available
    """
    try:
        import nibabel
        import numpy
        return True
    except ImportError:
        return False


def load_jnifti_file(filepath: Path) -> Dict[str, Any]:
    """
    Load JNIfTI file (either .jnii text or .bnii binary).
    
    Args:
        filepath: Path to .jnii or .bnii file
    
    Returns:
        JNIfTI data structure (dict with NIFTIHeader and NIFTIData)
    
    Raises:
        ValueError: If file format is invalid
        ImportError: If required library is missing
    """
    if filepath.suffix == '.jnii':
        # Text-based JSON format
        with open(filepath, 'r', encoding='utf-8') as f:
            jnii_data = json.load(f)
    
    elif filepath.suffix == '.bnii':
        # Binary JSON format (BJData/UBJSON)
        try:
            import bjdata
        except ImportError:
            raise ImportError(
                "bjdata library required for .bnii files. "
                "Install with: pip install bjdata"
            )
        
        with open(filepath, 'rb') as f:
            jnii_data = bjdata.load(f)
    
    else:
        raise ValueError(f"Unsupported file extension: {filepath.suffix}")
    
    # Validate structure
    if not isinstance(jnii_data, dict):
        raise ValueError("JNIfTI file must contain a JSON object")
    
    if 'NIFTIHeader' not in jnii_data or 'NIFTIData' not in jnii_data:
        raise ValueError(
            "Invalid JNIfTI structure: must have NIFTIHeader and NIFTIData fields"
        )
    
    return jnii_data


def jnii2nii(jnii_input: Union[str, Path, Dict], output_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Convert JNIfTI to NIfTI format.
    
    Pure Python implementation of MATLAB jnii2nii.m
    
    Args:
        jnii_input: JNIfTI file path or already-loaded dict structure
        output_path: If provided, save NIfTI to this path
    
    Returns:
        NIfTI structure dict with 'hdr' and 'img' fields
    """
    # Load JNIfTI data if input is a file path
    if isinstance(jnii_input, (str, Path)):
        jnii = load_jnifti_file(Path(jnii_input))
    else:
        jnii = jnii_input
    
    # Extract components
    jnii_header = jnii['NIFTIHeader']
    jnii_data = jnii['NIFTIData']
    
    # Handle JData array encoding
    if isinstance(jnii_data, dict) and '_ArrayData_' in jnii_data:
        img_array = np.array(jnii_data['_ArrayData_'])
    elif isinstance(jnii_data, list):
        img_array = np.array(jnii_data)
    elif isinstance(jnii_data, np.ndarray):
        img_array = jnii_data
    else:
        raise ValueError(f"Unsupported NIFTIData format: {type(jnii_data)}")
    
    # Determine NIfTI format (NIfTI-1 or NIfTI-2)
    nii_format = 'nifti1'
    if 'NIIFormat' in jnii_header:
        nii_format_str = jnii_header['NIIFormat']
        if nii_format_str.startswith(('ni2', 'n+2')):
            nii_format = 'nifti2'
    
    # Check if dimensions require NIfTI-2
    if 'Dim' in jnii_header and max(jnii_header['Dim']) >= 2**32:
        nii_format = 'nifti2'
    
    # Build affine matrix from JNIfTI header
    affine = _build_affine_from_jnifti_header(jnii_header)
    
    # Create NIfTI image
    if nii_format == 'nifti2':
        nifti_img = nib.Nifti2Image(img_array, affine)
    else:
        nifti_img = nib.Nifti1Image(img_array, affine)
    
    # Transfer metadata from JNIfTI header to NIfTI header
    _transfer_jnifti_metadata_to_nifti_header(nifti_img.header, jnii_header)
    
    # Save to file if output path provided
    if output_path:
        ensure_dir(output_path.parent)
        nib.save(nifti_img, str(output_path))
    
    # Return NIfTI structure (MATLAB-style)
    nii_struct = {
        'img': img_array,
        'hdr': nifti_img.header,
        'affine': affine
    }
    
    return nii_struct


def _build_affine_from_jnifti_header(jnii_header: Dict) -> np.ndarray:
    """
    Build 4x4 affine matrix from JNIfTI header.
    
    Priority:
    1. Use Affine matrix if available
    2. Use srow_x/y/z if available
    3. Build from qform (quaternion)
    4. Fallback to pixdim
    
    Args:
        jnii_header: JNIfTI NIFTIHeader dict
    
    Returns:
        4x4 affine matrix
    """
    # Priority 1: Direct affine matrix
    if 'Affine' in jnii_header:
        affine_data = jnii_header['Affine']
        
        if isinstance(affine_data, list):
            affine = np.array(affine_data, dtype=np.float64)
        else:
            affine = np.array(affine_data, dtype=np.float64)
        
        # Ensure 4x4 matrix
        if affine.shape == (3, 4):
            # Add bottom row [0, 0, 0, 1]
            affine = np.vstack([affine, [0, 0, 0, 1]])
        elif affine.shape == (4, 4):
            pass
        else:
            # Fallback
            affine = np.eye(4)
        
        return affine
    
    # Priority 2: Build from pixdim and qoffset
    affine = np.eye(4)
    
    if 'VoxelSize' in jnii_header:
        voxel_size = jnii_header['VoxelSize']
        if len(voxel_size) >= 3:
            affine[0, 0] = voxel_size[0]
            affine[1, 1] = voxel_size[1]
            affine[2, 2] = voxel_size[2]
    
    if 'QuaternOffset' in jnii_header:
        qoffset = jnii_header['QuaternOffset']
        if isinstance(qoffset, dict):
            affine[0, 3] = qoffset.get('x', 0)
            affine[1, 3] = qoffset.get('y', 0)
            affine[2, 3] = qoffset.get('z', 0)
    
    return affine


def _transfer_jnifti_metadata_to_nifti_header(nifti_header, jnii_header: Dict) -> None:
    """
    Transfer metadata from JNIfTI header to NIfTI header.
    
    Maps JNIfTI fields to corresponding NIfTI header fields.
    
    Args:
        nifti_header: nibabel NIfTI header object
        jnii_header: JNIfTI NIFTIHeader dict
    """
    # Dimension info
    if 'Dim' in jnii_header:
        dim = jnii_header['Dim']
        if isinstance(dim, list):
            dim_array = [len(dim)] + dim + [1] * (7 - len(dim))
            nifti_header['dim'] = dim_array[:8]
    
    # Data type
    if 'DataType' in jnii_header:
        dtype = jnii_header['DataType']
        if isinstance(dtype, str):
            dtype_code = NIFTI_DATATYPE_CODES.get(dtype.lower(), 0)
            if dtype_code > 0:
                nifti_header['datatype'] = dtype_code
    
    # Voxel size
    if 'VoxelSize' in jnii_header:
        voxel_size = jnii_header['VoxelSize']
        if isinstance(voxel_size, list):
            pixdim = [1.0] + voxel_size + [1.0] * (7 - len(voxel_size))
            nifti_header['pixdim'] = pixdim[:8]
    
    # Scaling
    if 'ScaleSlope' in jnii_header:
        nifti_header['scl_slope'] = jnii_header['ScaleSlope']
    
    if 'ScaleOffset' in jnii_header:
        nifti_header['scl_inter'] = jnii_header['ScaleOffset']
    
    # Intensity range
    if 'MaxIntensity' in jnii_header:
        nifti_header['cal_max'] = jnii_header['MaxIntensity']
    
    if 'MinIntensity' in jnii_header:
        nifti_header['cal_min'] = jnii_header['MinIntensity']
    
    # Timing
    if 'SliceTime' in jnii_header:
        nifti_header['slice_duration'] = jnii_header['SliceTime']
    
    if 'TimeOffset' in jnii_header:
        nifti_header['toffset'] = jnii_header['TimeOffset']
    
    # Description
    if 'Description' in jnii_header:
        desc = jnii_header['Description']
        if isinstance(desc, str):
            nifti_header['descrip'] = desc.encode('utf-8')[:80]
    
    # Intent
    if 'Intent' in jnii_header:
        intent = jnii_header['Intent']
        if isinstance(intent, str):
            intent_code = NIFTI_INTENT_CODES.get(intent.lower(), 0)
            nifti_header['intent_code'] = intent_code
    
    # Intent parameters
    if 'Param1' in jnii_header:
        nifti_header['intent_p1'] = jnii_header['Param1']
    if 'Param2' in jnii_header:
        nifti_header['intent_p2'] = jnii_header['Param2']
    if 'Param3' in jnii_header:
        nifti_header['intent_p3'] = jnii_header['Param3']
    
    # Slice info
    if 'FirstSliceID' in jnii_header:
        nifti_header['slice_start'] = jnii_header['FirstSliceID']
    
    if 'LastSliceID' in jnii_header:
        nifti_header['slice_end'] = jnii_header['LastSliceID']
    
    if 'SliceType' in jnii_header:
        slice_type = jnii_header['SliceType']
        if isinstance(slice_type, str):
            slice_code = NIFTI_SLICE_CODES.get(slice_type.lower(), 0)
            nifti_header['slice_code'] = slice_code
    
    # Units
    if 'Unit' in jnii_header:
        unit = jnii_header['Unit']
        if isinstance(unit, dict):
            space_unit = NIFTI_UNIT_CODES.get(unit.get('L', 'unknown'), 0)
            time_unit = NIFTI_UNIT_CODES.get(unit.get('T', 'unknown'), 0)
            nifti_header['xyzt_units'] = space_unit | time_unit
    
    # Transform codes
    if 'QForm' in jnii_header:
        qform = jnii_header['QForm']
        if isinstance(qform, str):
            qform_code = NIFTI_XFORM_CODES.get(qform.lower(), 0)
            nifti_header['qform_code'] = qform_code
    
    if 'SForm' in jnii_header:
        sform = jnii_header['SForm']
        if isinstance(sform, str):
            sform_code = NIFTI_XFORM_CODES.get(sform.lower(), 0)
            nifti_header['sform_code'] = sform_code
    
    # Quaternion parameters
    if 'Quatern' in jnii_header:
        quatern = jnii_header['Quatern']
        if isinstance(quatern, dict):
            nifti_header['quatern_b'] = quatern.get('b', 0)
            nifti_header['quatern_c'] = quatern.get('c', 0)
            nifti_header['quatern_d'] = quatern.get('d', 0)
    
    if 'QuaternOffset' in jnii_header:
        qoffset = jnii_header['QuaternOffset']
        if isinstance(qoffset, dict):
            nifti_header['qoffset_x'] = qoffset.get('x', 0)
            nifti_header['qoffset_y'] = qoffset.get('y', 0)
            nifti_header['qoffset_z'] = qoffset.get('z', 0)
    
    # Affine rows (srow_x/y/z)
    if 'Affine' in jnii_header:
        affine = np.array(jnii_header['Affine'])
        if affine.shape[0] >= 3:
            nifti_header['srow_x'] = affine[0, :4] if affine.shape[1] >= 4 else list(affine[0]) + [0]
            nifti_header['srow_y'] = affine[1, :4] if affine.shape[1] >= 4 else list(affine[1]) + [0]
            nifti_header['srow_z'] = affine[2, :4] if affine.shape[1] >= 4 else list(affine[2]) + [0]


def convert_jnifti_to_nifti(input_path: Path, output_path: Path, 
                            quiet: bool = False) -> Optional[Path]:
    """
    Convert JNIfTI file to NIfTI format.
    
    Main entry point for JNIfTI conversion.
    
    Args:
        input_path: Path to .jnii or .bnii file
        output_path: Output .nii.gz path
        quiet: Suppress output messages
    
    Returns:
        Path to created NIfTI file, or None if failed
    """
    if not check_jnifti_support():
        if not quiet:
            warn("nibabel not installed")
            warn("Install with: pip install nibabel")
        return None
    
    if not input_path.exists():
        if not quiet:
            warn(f"Input file not found: {input_path}")
        return None
    
    try:
        if not quiet:
            info(f"  Converting JNIfTI: {input_path.name}")
        
        # Load and convert
        nii_struct = jnii2nii(input_path, output_path)
        
        if not quiet:
            info(f"  ✓ Created: {output_path.name}")
            info(f"    Shape: {nii_struct['img'].shape}")
            info(f"    Dtype: {nii_struct['img'].dtype}")
        
        return output_path
        
    except ImportError as e:
        if not quiet:
            warn(f"  Missing dependency: {e}")
            if 'bjdata' in str(e):
                warn(f"  Install with: pip install bjdata")
        return None
    except Exception as e:
        if not quiet:
            warn(f"  JNIfTI conversion failed: {e}")
            import traceback
            traceback.print_exc()
        return None


def convert_jnifti_batch(jnifti_files: List[Path], output_dir: Path,
                        quiet: bool = False) -> List[Path]:
    """
    Batch convert multiple JNIfTI files to NIfTI format.
    
    Args:
        jnifti_files: List of .jnii or .bnii files
        output_dir: Output directory
        quiet: Suppress output
    
    Returns:
        List of successfully converted files
    """
    if not check_jnifti_support():
        if not quiet:
            warn("nibabel not installed for JNIfTI conversion")
        return []
    
    ensure_dir(output_dir)
    
    converted_files = []
    
    if not quiet:
        info(f"Converting {len(jnifti_files)} JNIfTI files...")
    
    for jnifti_file in jnifti_files:
        # Generate output filename
        output_name = jnifti_file.stem
        
        # Remove .jnii or .bnii if present in stem
        if output_name.endswith(('.jnii', '.bnii')):
            output_name = output_name.rsplit('.', 1)[0]
        
        output_path = output_dir / f"{output_name}.nii.gz"
        
        result = convert_jnifti_to_nifti(jnifti_file, output_path, quiet=True)
        
        if result:
            converted_files.append(result)
    
    if not quiet:
        info(f"✓ Converted {len(converted_files)}/{len(jnifti_files)} files")
    
    return converted_files


# ============================================================================
# Legacy: MATLAB Arrays → NIfTI
# ============================================================================

def arrays_to_nifti(final_plan: Dict[str, Any], input_root: Path,
                   output_path: Path) -> Path:
    """
    Convert MATLAB arrays to NIfTI based on final mapping plan.
    
    Args:
        final_plan: Final mapping plan from LLM
        input_root: Root directory of input data
        output_path: Output NIfTI file path
    
    Returns:
        Path to created NIfTI file
    """
    ensure_dir(output_path.parent)
    
    info(f"Converting MATLAB arrays to NIfTI: {output_path.name}")
    
    conversions = final_plan.get("conversions", [])
    if not conversions:
        warn("No conversions specified in final_plan")
        return output_path
    
    conversion = conversions[0]
    
    source_file = conversion.get("source_file", "")
    source_var = conversion.get("source_variable", "")
    operations = conversion.get("operations", [])
    sidecar = conversion.get("sidecar", {})
    
    mat_path = input_root / source_file
    
    if not mat_path.exists():
        warn(f"MAT file not found: {mat_path}")
        return output_path
    
    info(f"  Loading: {mat_path}")
    info(f"  Variable: {source_var}")
    
    try:
        mat_data = loadmat(str(mat_path))
        
        if source_var not in mat_data:
            available = [k for k in mat_data.keys() if not k.startswith('__')]
            warn(f"Variable '{source_var}' not found. Available: {available}")
            return output_path
        
        img_data = mat_data[source_var]
        info(f"  Loaded shape: {img_data.shape}, dtype: {img_data.dtype}")
        
        affine = np.eye(4)
        
        for op in operations:
            op_type = op.get("type")
            
            if op_type == "transpose":
                axes = op.get("axes", [0, 1, 2])
                info(f"  Transposing: {axes}")
                img_data = np.transpose(img_data, axes)
            
            elif op_type == "flip":
                flip_axes = op.get("axes", [])
                for axis in flip_axes:
                    info(f"  Flipping axis: {axis}")
                    img_data = np.flip(img_data, axis=axis)
            
            elif op_type == "build_affine":
                voxel_size = op.get("voxel_size_mm", [1.0, 1.0, 1.0])
                coord_sys = op.get("coordinate_system", "RAS")
                
                info(f"  Building affine: voxel={voxel_size}, coord={coord_sys}")
                
                affine = np.eye(4)
                affine[0, 0] = voxel_size[0]
                affine[1, 1] = voxel_size[1]
                affine[2, 2] = voxel_size[2]
                
                if coord_sys == "LPS":
                    affine[0, 0] = -affine[0, 0]
                    affine[1, 1] = -affine[1, 1]
        
        nifti_img = nib.Nifti1Image(img_data, affine)
        
        tr = sidecar.get("RepetitionTime")
        if tr and len(img_data.shape) > 3:
            zooms = list(nifti_img.header.get_zooms()[:3]) + [tr]
            nifti_img.header.set_zooms(zooms)
            info(f"  Set TR: {tr}s")
        
        nib.save(nifti_img, str(output_path))
        info(f"  ✓ Created NIfTI: {output_path}")
        
        sidecar_path = output_path.parent / (output_path.stem.replace('.nii', '') + '.json')
        write_json(sidecar_path, sidecar)
        info(f"  ✓ Created sidecar: {sidecar_path.name}")
        
        return output_path
        
    except Exception as e:
        warn(f"Failed to convert arrays to NIfTI: {e}")
        import traceback
        traceback.print_exc()
        return output_path


def validate_nifti(nifti_path: Path) -> bool:
    """
    Validate NIfTI file integrity.
    
    Args:
        nifti_path: Path to NIfTI file
    
    Returns:
        True if valid, False otherwise
    """
    try:
        img = nib.load(str(nifti_path))
        shape = img.shape
        dtype = img.get_data_dtype()
        
        info(f"NIfTI valid: {nifti_path.name} - shape {shape}, dtype {dtype}")
        return True
        
    except Exception as e:
        warn(f"NIfTI validation failed for {nifti_path}: {e}")
        return False
