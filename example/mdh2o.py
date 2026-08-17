"""Short H2O MD example using OpenMM and openmmpyscf.PySCFPotential."""

from pathlib import Path
import sys

import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit

# Allow running this script directly from the example folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openmmpyscf import PySCFPotential


def build_h2o_system(method="b3lyp", basis="6-31g*"):
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("HOH", chain)
    topology.addAtom("O", app.element.oxygen, residue)
    topology.addAtom("H1", app.element.hydrogen, residue)
    topology.addAtom("H2", app.element.hydrogen, residue)

    potential = PySCFPotential(
        method=method,
        basis=basis,
        charge=0,
        multiplicity=1,
        memory="1 GB",
        num_threads=1,
    )
    system = potential.createSystem(topology)

    positions = np.array(
        [
            [0.000, 0.000, 0.000],
            [0.096, 0.000, 0.000],
            [-0.024, 0.093, 0.000],
        ],
        dtype=float,
    ) * unit.nanometer

    return system, positions


def run_short_md(steps=10):
    system, positions = build_h2o_system()

    integrator = mm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.5 * unit.femtosecond,
    )

    context = mm.Context(system, integrator)
    context.setPositions(positions)

    state0 = context.getState(getEnergy=True, getForces=True)
    print("Initial energy:", state0.getPotentialEnergy())
    print("Initial forces (kJ/mol/nm):")
    print(state0.getForces(asNumpy=True))

    integrator.step(int(steps))

    state1 = context.getState(getEnergy=True)
    print(f"Energy after {steps} steps:", state1.getPotentialEnergy())


if __name__ == "__main__":
    run_short_md(steps=30)