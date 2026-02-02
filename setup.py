#!/usr/bin/env python3
"""
Setup script for autobidsify (flat layout).
"""

from setuptools import setup, find_packages
from pathlib import Path


def get_version():
    """Extract version from __init__.py."""
    init_file = Path(__file__).parent / "autobidsify" / "__init__.py"
    if init_file.exists():
        for line in init_file.read_text().splitlines():
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.5.0"


def get_long_description():
    """Read README.md."""
    readme_file = Path(__file__).parent / "README.md"
    if readme_file.exists():
        return readme_file.read_text(encoding="utf-8")
    return ""


setup(
    name="autobidsify",
    version=get_version(),
    author="Yiyi Liu",
    author_email="yiyi.liu3@northeastern.edu",
    description="Automated BIDS standardization tool powered by LLM-first architecture",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/fangzhouliucode/autobidsify",
    project_urls={
        "Documentation": "https://autobidsify.readthedocs.io",
        "Repository": "https://github.com/fangzhouliucode/autobidsify",
        "Issues": "https://github.com/fangzhouliucode/autobidsify/issues",
    },
    license="MIT",
    
    # Flat layout - no package_dir needed
    packages=find_packages(exclude=["tests", "tests.*", "docs"]),
    
    python_requires=">=3.10",
    
    install_requires=[
        "openai>=1.0.0",
        "pyyaml>=6.0",
        "pdfplumber>=0.10.0",
        "PyPDF2>=3.0.0",
        "python-docx>=1.0.0",
        "pydicom>=2.4.0",
        "nibabel>=5.0.0",
        "h5py>=3.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "openpyxl>=3.1.0",
    ],
    
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "ruff>=0.1.0",
            "mypy>=1.0",
        ],
    },
    
    entry_points={
        "console_scripts": [
            "autobidsify=autobidsify.__main__:main",
            "bidsify=autobidsify.__main__:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    
    include_package_data=True,
    zip_safe=False,
)
```

## Final Flat Layout Structure
```
autobidsify/                    # Repository root
├── pyproject.toml
├── setup.py
├── README.md
├── LICENSE
├── CHANGELOG.md
├── .gitignore
│
├── autobidsify/                # Package directory
│   ├── __init__.py
│   ├── __main__.py
│   ├── constants.py
│   ├── filename_tokenizer.py
│   ├── llm.py
│   ├── universal_core.py
│   ├── utils.py
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── classification.py
│   │   ├── evidence.py
│   │   ├── ingest.py
│   │   └── trio.py
│   └── converters/
│       ├── __init__.py
│       ├── executor.py
│       ├── jnifti_converter.py
│       ├── mri_convert.py
│       ├── nirs_convert.py
│       ├── planner.py
│       └── validators.py
│
├── tests/
│   └── __init__.py
│
└── docs/