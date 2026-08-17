"""Periodic QM/MM ACE+water example using OpenMM and openmm-pyscf.

Workflow:
1. Load examples/data/acewater.pdb.
2. Build an MM reference system with Amber + SPC water, with OpenFF 2.3
    templates for chain Q residues.
3. Define the QM region as all atoms in chain Q.
4. Create a periodic mixed QM/MM system with PySCF electronic embedding.
5. Add a flat-bottom quadratic restraint between ACE C1 and WQM O.
6. Minimize for up to 50 iterations.
7. Assign velocities at 50 K and run 100 NVT Langevin steps with a 0.5 fs timestep.
"""

from pathlib import Path
import sys

import openmm as mm
import openmm.app as app
from openmm import unit

# Allow running this script directly from the examples folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openmmpyscf import PySCFPotential


def _get_ace_wqm_restraint_atoms(topology):
    """Return atom indices for ACE carboxy carbon (C1) and WQM oxygen (O) in chain Q."""
    ace_c1 = None
    wqm_o = None

    for atom in topology.atoms():
        residue = atom.residue
        if residue.chain.id != "Q":
            continue
        if residue.name == "ACE" and atom.name == "C1":
            ace_c1 = atom.index
        elif residue.name == "WQM" and atom.name == "O":
            wqm_o = atom.index

    if ace_c1 is None or wqm_o is None:
        raise RuntimeError(
            "Could not find ACE:C1 and WQM:O atoms in chain Q for restraint"
        )

    return ace_c1, wqm_o


def _add_flat_bottom_restraint(system, atom1, atom2):
    """Add a flat-bottom quadratic restraint active only for r > 4.5 Angstrom."""
    restraint = mm.CustomBondForce("0.5*k*step(r-r0)*(r-r0)^2")
    restraint.addGlobalParameter("r0", 0.45)  # nm (4.5 Angstrom)
    k_kcal_per_mol_a2 = 25.0 * unit.kilocalorie_per_mole / unit.angstrom**2
    k_kj_per_mol_nm2 = k_kcal_per_mol_a2.value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer**2
    )
    restraint.addGlobalParameter("k", k_kj_per_mol_nm2)
    restraint.addBond(int(atom1), int(atom2), [])
    system.addForce(restraint)


def _register_chain_q_templates(forcefield, topology):
    """Register OpenFF templates for nonstandard residues in chain Q."""
    try:
        from openff.toolkit import Molecule
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    except Exception as exc:
        raise RuntimeError(
            "This example requires openmmforcefields and openff-toolkit to parameterize "
            "chain Q residues (ACE/WQM)."
        ) from exc

    residue_smiles = {
        "ACE": "CC(=O)O",
        "WQM": "O",
    }

    chain_q_names = {
        residue.name
        for residue in topology.residues()
        if residue.chain.id == "Q"
    }

    unsupported = sorted(name for name in chain_q_names if name not in residue_smiles)
    if unsupported:
        raise RuntimeError(
            "Unsupported chain Q residue names for this example: "
            + ", ".join(unsupported)
        )

    molecules = []
    for residue_name in sorted(chain_q_names):
        molecule = Molecule.from_smiles(residue_smiles[residue_name])
        molecule.name = residue_name
        molecules.append(molecule)

    smirnoff = SMIRNOFFTemplateGenerator(
        molecules=molecules,
        forcefield="openff-2.3.0",
    )
    forcefield.registerTemplateGenerator(smirnoff.generator)


def build_mixed_system():
    pdb_path = Path(__file__).resolve().parent / "data" / "acewater.pdb"
    pdb = app.PDBFile(str(pdb_path))

    # Amber protein parameters + SPC/E water model.
    forcefield = app.ForceField("amber14-all.xml", "amber14/spce.xml")
    _register_chain_q_templates(forcefield, pdb.topology)
    mm_system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
    )

    # Chain Q defines the QM region. This should include complete molecules only.
    qm_atoms = [atom.index for atom in pdb.topology.atoms() if atom.residue.chain.id == "Q"]
    if not qm_atoms:
        raise RuntimeError("No QM atoms found: chain 'Q' is missing in acewater.pdb")

    potential = PySCFPotential(
        method="wb97x-d4",
        basis="def2-TZVP",
        charge=0,
        multiplicity=1,
        memory="2 GB",
        num_threads=None,
        quiet=True,
    )

    mixed_system = potential.createMixedSystem(
        pdb.topology,
        mm_system,
        atoms=qm_atoms,
        embedding="electronic",
        forceGroup=4,
    )

    ace_c1, wqm_o = _get_ace_wqm_restraint_atoms(pdb.topology)
    _add_flat_bottom_restraint(mixed_system, ace_c1, wqm_o)

    if mixed_system.getDefaultPeriodicBoxVectors() is None:
        raise RuntimeError("Mixed system is expected to preserve periodic box vectors")

    return pdb, mm_system, mixed_system, qm_atoms


def _create_integrator():
    return mm.LangevinMiddleIntegrator(
        50.0 * unit.kelvin,
        0.5 / unit.picosecond,
        0.5 * unit.femtosecond,
    )


def _write_pdb_snapshot(path, topology, positions):
    with open(path, "w") as handle:
        app.PDBFile.writeFile(topology, positions, handle)


def _run_mm_minimization(pdb, mm_system, platform, output_dir):
    integrator = _create_integrator()
    simulation = app.Simulation(pdb.topology, mm_system, integrator, platform)
    simulation.context.setPositions(pdb.positions)

    state0 = simulation.context.getState(getEnergy=True)
    print("Initial MM potential energy:", state0.getPotentialEnergy())

    simulation.minimizeEnergy(maxIterations=50)
    minimized_state = simulation.context.getState(
        getEnergy=True,
        getPositions=True,
    )
    print(
        "Post-MM minimization potential energy:",
        minimized_state.getPotentialEnergy(),
    )

    mm_pdb_path = output_dir / "mdacewater_mm_minimized.pdb"
    _write_pdb_snapshot(mm_pdb_path, pdb.topology, minimized_state.getPositions())
    print(f"Wrote MM-minimized structure to {mm_pdb_path}")

    return minimized_state.getPositions()


def _run_qmmm_minimization_and_md(pdb, mixed_system, platform, positions, output_dir):
    integrator = _create_integrator()
    simulation = app.Simulation(pdb.topology, mixed_system, integrator, platform)
    simulation.context.setPositions(positions)

    state_qmmm0 = simulation.context.getState(getEnergy=True)
    print("Initial QM/MM potential energy:", state_qmmm0.getPotentialEnergy())

    simulation.minimizeEnergy(maxIterations=50)
    qmmm_minimized_state = simulation.context.getState(
        getEnergy=True,
        getPositions=True,
    )
    print(
        "Post-QM/MM minimization potential energy:",
        qmmm_minimized_state.getPotentialEnergy(),
    )

    qmmm_min_pdb_path = output_dir / "mdacewater_qmmm_minimized.pdb"
    _write_pdb_snapshot(
        qmmm_min_pdb_path,
        pdb.topology,
        qmmm_minimized_state.getPositions(),
    )
    print(f"Wrote QM/MM-minimized structure to {qmmm_min_pdb_path}")

    simulation.context.setVelocitiesToTemperature(50.0 * unit.kelvin)

    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout,
            5,
            step=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
        )
    )

    simulation.step(100)
    state_final = simulation.context.getState(getEnergy=True, getPositions=True)
    print("Potential energy after 100 NVT steps:", state_final.getPotentialEnergy())

    final_pdb_path = output_dir / "mdacewater_qmmm_final.pdb"
    _write_pdb_snapshot(final_pdb_path, pdb.topology, state_final.getPositions())
    print(f"Wrote final QM/MM structure to {final_pdb_path}")


def run_mdacewater():
    pdb, mm_system, mixed_system, qm_atoms = build_mixed_system()

    platform = mm.Platform.getPlatformByName("CPU")
    print(f"QM atom count (chain Q): {len(qm_atoms)}")

    output_dir = Path(__file__).resolve().parent
    mm_positions = _run_mm_minimization(pdb, mm_system, platform, output_dir)
    _run_qmmm_minimization_and_md(
        pdb,
        mixed_system,
        platform,
        mm_positions,
        output_dir,
    )


if __name__ == "__main__":
    run_mdacewater()