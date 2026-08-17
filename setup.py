from setuptools import find_packages, setup


setup(
    name="openmm-pyscf",
    version="0.1.0",
    description="OpenMM-PySCF integration via OpenMM PythonForce callbacks",
    packages=find_packages(exclude=("test", "test.*", "example", "example.*", "examples", "examples.*")),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "openmm",
        "pyscf",
    ],
)