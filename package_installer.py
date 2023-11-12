import sys
import subprocess

lst_package = [
    sys.executable,
    "-m",
    "pip",
    "install",
    "pandas",
    "numpy",
    "matplotlib",
    "datetime",
    "sklearn",
    "scikit-learn",
    'openpyxl',
    "--index-url=https://pypi.org/simple", # use the default PyPI index
    "--extra-index-url=https://bcms.bloomberg.com/pip/simple", # use the Bloomberg index as a backup
    "blpapi",
    "tkcalendar"
]

try:
    reqs = subprocess.check_output(lst_package)
    installed_packages = [r.decode().split("==")[0] for r in reqs.split()]
    print("Successfully installed packages: ", installed_packages)
except subprocess.CalledProcessError as e:
    print("Failed to install packages. Error message: ", e.output.decode())
