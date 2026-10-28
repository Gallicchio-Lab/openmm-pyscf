# OpenMM-PySCF

`openmm-pyscf` integrates PySCF electronic-structure calculations into OpenMM through `openmm.PythonForce` callbacks.

The API mirrors the [openmm-ml](https://github.com/openmm/openmm-ml) package. It exposes three public symbols:

- `PySCFPotential`
- `PySCFPythonForce`
- `make_openmm_python_force`

## Use Cases

- Simulate systems at the QM and QM/MM levels of theory
- Compare the accuracy of ML potentials (available through [openmm-ml](https://github.com/openmm/openmm-ml)) to QM and QM/MM references

## Credits

- Author: Emilio Gallicchio <emilio.gallicchio@gmail.com>
- The [OpenMM](https://openmm.org/development) development team
- The [PySCF](https://pyscf.org/about.html) development team
- This software, including tests, examples, and documentation, was developed with the assistance of an AI coding agent (Gemini 3.7 Flash)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Installation

### Prerequisites

`openmm-pyscf` requires recent versions of:
- **`openmm`**
- **`pyscf`**
- **`numpy`**
- **`pyscf-dispersion`** (required for D3/D4 dispersion-corrected functionals, e.g. `b3lyp-d3`, `pbe0-d3`, etc.)

Optional packages for testing and running the examples:
- **`pytest`** (for running the test suite)
- **`openmmforcefields`** and **`openff-toolkit`** (for running QM/MM examples with OpenFF/Amber force fields)

### Creating a Conda Environment

You can install all dependencies from `conda-forge` using the provided `requirements.txt`:

```bash
# Create a new conda environment and install dependencies
conda create -n openmm-pyscf --file requirements.txt -c conda-forge

# Activate the environment
conda activate openmm-pyscf
```

Alternatively, to install dependencies into an existing active environment:

```bash
conda install --file requirements.txt -c conda-forge
```

### Installing `openmm-pyscf`

Install `openmm-pyscf` from the repository root:

```bash
pip install .
```

## Public API

### `PySCFPotential`

`PySCFPotential` is the main user-facing entry point. It stores the PySCF settings once, then builds either:

- a full-QM OpenMM `System` with `createSystem(topology)`, or
- a QM/MM mixed `System` with `createMixedSystem(topology, system, atoms, ...)` using electronic embedding.
- see the [PySCF documentation](https://pyscf.org/user/index.html) for the meaning and valid settings for the QM model parameters (`method`, `basis`, etc.)

Constructor:

```python
PySCFPotential(
    method="b3lyp",
    basis="3-21g",
    charge=0,
    multiplicity=1,
    memory="1 GB",
    num_threads=None,
    pyscf_options=None,
    solvation_model=None,
    solvation_options=None,
    density_fit=True,
    use_gpu=False,
    quiet=True,
    rcut_ewald=None,
    rcut_hcore=None,
    mm_radii=None,
)
```

Parameters:

- `method`: PySCF method or DFT functional string. Examples: `"hf"`, `"rhf"`, `"uhf"`, `"rohf"`, `"b3lyp"`.
- `basis`: PySCF basis name.
- `charge`: Total QM charge.
- `multiplicity`: Spin multiplicity. Internally, PySCF uses `spin = multiplicity - 1`.
- `memory`: PySCF memory limit. Accepts values like `1024`, `"1024"`, or `"1 GB"`.
- `num_threads`: Optional PySCF thread count.
- `pyscf_options`: Optional mapping of attributes to set directly on the PySCF mean-field object.
- `solvation_model`: Optional implicit-solvent model for full-QM calculations. Supported helper names currently include `ddcosmo`, `pcm`, and `smd`.
- `solvation_options`: Optional mapping of attributes to set on the solvent-wrapped PySCF object.
- `density_fit`: If `True` (default), uses PySCF density fitting for the QM calculation. Set to `False` to disable it.
- `use_gpu`: If `True`, attempts to convert the PySCF method to GPU4PySCF when available.
- `quiet`: If `True`, uses low PySCF verbosity.
- `rcut_ewald`: Optional real-space cutoff (in Å) for Ewald summation in periodic QM/MM (defaults to half the minimum box dimension).
- `rcut_hcore`: Optional cutoff (in Å) for exact QM-MM core Hamiltonian coupling in periodic QM/MM (defaults to half the minimum box dimension).
- `mm_radii`: Optional Gaussian charge distribution radii (in Å) for MM atoms in periodic QM/MM (defaults to `1.0` Å). Can be a scalar or an array/list of length equal to the number of MM atoms.

Methods:

- `createSystem(topology, removeCMMotion=True) -> openmm.System`
- `createMixedSystem(topology, system, atoms, removeConstraints=True, forceGroup=0, interpolate=False, embedding="electronic", **args) -> openmm.System`
- `createCalculator(topology) -> PySCFPythonForce`
- `getSupportedEmbeddings() -> list[str]`

#### `createSystem(topology)`

Builds a full-QM OpenMM system. Each atom mass is copied from the OpenMM topology, and the PySCF energy/force evaluation is attached as a Python force.

Current behavior:

- Full-QM systems are non-periodic.
- Implicit solvation is supported here.

#### `createMixedSystem(topology, system, atoms, ...)`

Builds a QM/MM system using electronic (electrostatic) embedding.

Arguments:

- `topology`: OpenMM topology for the full system.
- `system`: Reference MM OpenMM system.
- `atoms`: Atom indices assigned to the QM region.
- `removeConstraints`: If `True`, removes constraints that lie entirely inside the QM region.
- `forceGroup`: Force group assigned to the QM Python force.
- `interpolate`: Currently unsupported and must remain `False`.
- `embedding`: Must be `"electronic"`, which refers to the MM charges embedding scheme in PySCF.
- `periodicBoxVectors` (or `periodic_box_vectors`): Optional `(3, 3)` periodic box vectors. If omitted, inferred from the topology or system if periodic boundary conditions are active.
- `rcut_ewald`: Optional override for the Ewald real-space cutoff (Å) in periodic QM/MM.
- `rcut_hcore`: Optional override for the exact coupling cutoff (Å) in periodic QM/MM.
- `mm_radii` (or `radii`): Optional override for the MM Gaussian charge distribution radii (Å).

Current behavior and limitations:

- Requires exactly one `openmm.NonbondedForce` in the MM reference system.
- Removes bonded terms involving QM atoms from the MM system copy.
- Electrostatic interactions between QM and MM atoms are evaluated quantum mechanically via electronic embedding, while QM-QM and QM-MM electrostatic interactions in the MM `NonbondedForce` are zeroed out via exceptions.
- Lennard-Jones interactions between QM-MM and MM-MM atoms are retained in OpenMM's `NonbondedForce`.
- QM region selection must include whole molecules. Splitting a covalently bonded molecule across the QM/MM boundary (dangling QM bonds) is not supported.
- **Periodic Boundary Conditions (PBC)**:
  - When periodic box vectors are present, periodic QM/MM is evaluated using PySCF's `pyscf.qmmm.pbc` interface (`itrf.add_mm_charges`), with Gaussian MM charge distributions and periodic Ewald summation.
  - The simulation box must be orthogonal (diagonal). Non-orthogonal boxes will raise a `ValueError`.
- Implicit solvation is rejected for QM/MM. `createMixedSystem()` raises `ValueError` if `solvation_model` is set.

### `PySCFPythonForce`

`PySCFPythonForce` is a lower-level callback wrapper around a PySCF calculation. It is useful when you want direct access to the OpenMM callback object or want to evaluate energies and forces from explicit positions.

Constructor:

```python
PySCFPythonForce(
    symbols,
    method="b3lyp",
    basis="3-21g",
    charge=0,
    multiplicity=1,
    memory="1 GB",
    num_threads=None,
    pyscf_options=None,
    solvation_model=None,
    solvation_options=None,
    density_fit=True,
    use_gpu=False,
    quiet=True,
)
```

Important methods:

- `compute(state)`
- `energy_and_forces(positions)`

`compute(state)` is the callback used by OpenMM PythonForce.

`energy_and_forces(positions)` returns a `PySCFResult` dataclass containing:

- `energy`: OpenMM quantity in `kilojoules_per_mole`
- `forces`: OpenMM quantity in `kilojoules_per_mole / nanometer`

### `make_openmm_python_force`

```python
make_openmm_python_force(pyscf_force: PySCFPythonForce) -> openmm.Force
```

Wraps a `PySCFPythonForce` instance in an OpenMM `PythonForce`.

## Example: full-QM water

```python
import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

from openmmpyscf import PySCFPotential

topology = app.Topology()
chain = topology.addChain()
residue = topology.addResidue("HOH", chain)
topology.addAtom("O", app.element.oxygen, residue)
topology.addAtom("H1", app.element.hydrogen, residue)
topology.addAtom("H2", app.element.hydrogen, residue)

potential = PySCFPotential(method="b3lyp", basis="6-31g*", num_threads=1)
system = potential.createSystem(topology)

positions = np.array([
    [0.000, 0.000, 0.000],
    [0.096, 0.000, 0.000],
    [-0.024, 0.093, 0.000],
]) * unit.nanometer

integrator = mm.VerletIntegrator(0.5 * unit.femtosecond)
context = mm.Context(system, integrator)
context.setPositions(positions)
state = context.getState(getEnergy=True, getForces=True)
print(state.getPotentialEnergy())
print(state.getForces(asNumpy=True))
```

See also [examples/mdh2o.py](examples/mdh2o.py) for a short MD example.

## Examples

- [examples/mdh2o.py](examples/mdh2o.py): full-QM one-water MD example.
- [examples/mdacewater.py](examples/mdacewater.py): periodic QM/MM ACE+water example with chain `Q` as the QM region, Amber+SPC water MM model, 50-step minimization, and 100 Langevin NVT steps.

## Example: QM/MM mixed system

```python
mixed_system = potential.createMixedSystem(
    topology,
    mm_system,
    atoms=[0, 1, 2],
    forceGroup=4,
    embedding="electronic",
)
```

In this mode:

- atoms in `atoms` are treated by PySCF,
- the remaining atoms remain in the MM system,
- MM charges polarize the QM calculation through electronic embedding (using `pyscf.qmmm.pbc` under PBC).

## Supported workflows

- Full-QM OpenMM energy and force evaluation
- Full-QM short MD using OpenMM integrators
- QM/MM electronic embedding (non-periodic and periodic via `pyscf.qmmm.pbc`)
- Periodic QM/MM with Ewald electrostatics
- Implicit solvation for full-QM calculations only

## Unsupported or restricted behavior

- Full-QM periodic PySCF calculations are not part of the public API.
- QM/MM only supports `embedding="electronic"`.
- QM/MM with `solvation_model` is rejected.
- Periodic QM/MM requires an orthogonal simulation box.
- `interpolate=True` in `createMixedSystem()` is not supported.
- Multiple `NonbondedForce` objects in the MM reference system are not supported.

## Notes on units

- Input positions are expected in nanometers when passed through OpenMM states or OpenMM quantities.
- Internal PySCF geometries are built in Angstroms.
- Reported energies are converted to `kJ/mol`.
- Reported forces are converted to `kJ/mol/nm`.

