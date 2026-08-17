import numpy as np
import openmm as mm
import openmm.app as app
from openmm import unit
import pytest
from openmmpyscf import PySCFPotential, PySCFPythonForce


def create_pyscf_openmm_one_water_system():
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("HOH", chain)
    topology.addAtom("O", app.element.oxygen, residue)
    topology.addAtom("H1", app.element.hydrogen, residue)
    topology.addAtom("H2", app.element.hydrogen, residue)

    potential = PySCFPotential(
        method="b3lyp",
        basis="6-31g*",
        charge=0,
        multiplicity=1,
        memory="1 GB",
    )
    system = potential.createSystem(topology)

    initial_positions = np.array(
        [
            [0.000, 0.000, 0.000],
            [0.096, 0.000, 0.000],
            [-0.024, 0.093, 0.000],
        ]
    ) * unit.nanometer

    return system, initial_positions


def create_pyscf_opemm_two_water_system(periodic_box_vectors=None, **mixed_args):
    """Build full-QM and mixed QM/MM two-water systems plus shared positions."""
    topology = app.Topology()
    chain = topology.addChain()

    if periodic_box_vectors is not None:
        topology.setPeriodicBoxVectors(periodic_box_vectors)

    residue_qm = topology.addResidue("HOH", chain)
    topology.addAtom("O1", app.element.oxygen, residue_qm)
    topology.addAtom("H11", app.element.hydrogen, residue_qm)
    topology.addAtom("H12", app.element.hydrogen, residue_qm)

    residue_mm = topology.addResidue("HOH", chain)
    topology.addAtom("O2", app.element.oxygen, residue_mm)
    topology.addAtom("H21", app.element.hydrogen, residue_mm)
    topology.addAtom("H22", app.element.hydrogen, residue_mm)

    positions = np.array(
        [
            [0.000, 0.000, 0.000],
            [0.0957, 0.000, 0.000],
            [-0.0239, 0.0927, 0.000],
            [0.280, 0.000, 0.000],
            [0.3757, 0.000, 0.000],
            [0.2561, 0.0927, 0.000],
        ]
    ) * unit.nanometer

    potential = PySCFPotential(
        method="b3lyp",
        basis="3-21g",
        charge=0,
        multiplicity=1,
        memory="1 GB",
        num_threads=1,
    )

    full_qm_system = potential.createSystem(topology)

    mm_system = mm.System()
    for atom in topology.atoms():
        mm_system.addParticle(atom.element.mass)

    bond_force = mm.HarmonicBondForce()
    angle_force = mm.HarmonicAngleForce()
    nonbonded = mm.NonbondedForce()

    oh_length = 0.1 * unit.nanometer
    hoh_angle = 109.47 * unit.degrees
    k_bond = 345000 * unit.kilojoules_per_mole / unit.nanometer**2
    k_angle = 383 * unit.kilojoules_per_mole / unit.radian**2

    for base in (0, 3):
        o, h1, h2 = base, base + 1, base + 2
        bond_force.addBond(o, h1, oh_length, k_bond)
        bond_force.addBond(o, h2, oh_length, k_bond)
        angle_force.addAngle(h1, o, h2, hoh_angle, k_angle)

    mm_system.addForce(bond_force)
    mm_system.addForce(angle_force)

    charge_o = -0.82
    charge_h = 0.41
    sigma_o = 0.316557 * unit.nanometer
    epsilon_o = 0.6502 * unit.kilojoules_per_mole
    sigma_h = 1.0 * unit.nanometer
    epsilon_h = 0.0 * unit.kilojoules_per_mole

    for atom_index in range(6):
        if atom_index in (0, 3):
            nonbonded.addParticle(charge_o, sigma_o, epsilon_o)
        else:
            nonbonded.addParticle(charge_h, sigma_h, epsilon_h)

    for base in (0, 3):
        o, h1, h2 = base, base + 1, base + 2
        nonbonded.addException(o, h1, 0.0, sigma_h, epsilon_h)
        nonbonded.addException(o, h2, 0.0, sigma_h, epsilon_h)
        nonbonded.addException(h1, h2, 0.0, sigma_h, epsilon_h)

    mm_system.addForce(nonbonded)

    if periodic_box_vectors is not None:
        mm_system.setDefaultPeriodicBoxVectors(*periodic_box_vectors)

    mixed_system = potential.createMixedSystem(
        topology,
        mm_system,
        atoms=[0, 1, 2],
        embedding="electronic",
        **mixed_args,
    )

    return full_qm_system, mixed_system, positions


def report_system_state(system: mm.System, positions):
    """Evaluate potential energy and forces for an OpenMM System."""
    integrator = mm.VerletIntegrator(0.5 * unit.femtosecond)
    context = mm.Context(system, integrator)
    try:
        context.setPositions(positions)
        state = context.getState(getEnergy=True, getForces=True)
        energy_kj_mol = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        forces_kj_mol_nm = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer
        )
        return energy_kj_mol, forces_kj_mol_nm
    finally:
        del context
        del integrator


def test_wrapper():
    """Sanity check on pure PySCF force wrapper."""
    symbols = ["O", "H", "H"]
    force = PySCFPythonForce(
        symbols=symbols,
        method="b3lyp",
        basis="6-31g*",
        charge=0,
        multiplicity=1,
        memory="1 GB",
    )

    initial_positions_nm = np.array(
        [
            [0.000, 0.000, 0.000],
            [0.096, 0.000, 0.000],
            [-0.024, 0.093, 0.000],
        ]
    )
    result = force.energy_and_forces(initial_positions_nm)
    assert result.energy.unit.is_compatible(unit.kilojoules_per_mole)
    assert result.forces.unit.is_compatible(unit.kilojoules_per_mole / unit.nanometer)


def test_solvation_configuration():
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("HOH", chain)
    topology.addAtom("O", app.element.oxygen, residue)
    topology.addAtom("H1", app.element.hydrogen, residue)
    topology.addAtom("H2", app.element.hydrogen, residue)

    potential = PySCFPotential(
        method="b3lyp",
        basis="6-31g*",
        charge=0,
        multiplicity=1,
        memory="1 GB",
        solvation_model="ddcosmo",
        solvation_options={"lebedev_order": 17},
    )
    calc = potential.createCalculator(topology)
    assert calc.solvation_model == "ddcosmo"
    assert calc.solvation_options == {"lebedev_order": 17}


def test_mixed_system_api():
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("HOH", chain)
    topology.addAtom("O", app.element.oxygen, residue)
    topology.addAtom("H1", app.element.hydrogen, residue)
    topology.addAtom("H2", app.element.hydrogen, residue)

    system = mm.System()
    for atom in topology.atoms():
        system.addParticle(atom.element.mass)

    nonbonded = mm.NonbondedForce()
    for _ in topology.atoms():
        nonbonded.addParticle(0.0, 1.0, 0.0)
    system.addForce(nonbonded)

    potential = PySCFPotential(
        method="b3lyp",
        basis="6-31g*",
        charge=0,
        multiplicity=1,
        memory="1 GB",
    )
    mixed_system = potential.createMixedSystem(
        topology,
        system,
        atoms=[0, 1, 2],
        forceGroup=3,
        embedding="electronic",
    )
    assert mixed_system.getNumForces() == system.getNumForces() + 1
    assert mixed_system.getForce(mixed_system.getNumForces() - 1).getForceGroup() == 3


def test_mixed_system_rejects_solvation():
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("HOH", chain)
    topology.addAtom("O", app.element.oxygen, residue)
    topology.addAtom("H1", app.element.hydrogen, residue)
    topology.addAtom("H2", app.element.hydrogen, residue)

    system = mm.System()
    for atom in topology.atoms():
        system.addParticle(atom.element.mass)

    nonbonded = mm.NonbondedForce()
    for _ in topology.atoms():
        nonbonded.addParticle(0.0, 1.0, 0.0)
    system.addForce(nonbonded)

    potential = PySCFPotential(
        method="b3lyp",
        basis="6-31g*",
        charge=0,
        multiplicity=1,
        memory="1 GB",
        solvation_model="ddcosmo",
    )
    with pytest.raises(ValueError, match="solvation_model"):
        potential.createMixedSystem(
            topology,
            system,
            atoms=[0, 1, 2],
            embedding="electronic",
        )


def test_two_waters_full_qm_and_mixed():
    full_qm_system, mixed_system, positions = create_pyscf_opemm_two_water_system()
    energy_full, forces_full = report_system_state(full_qm_system, positions)
    energy_mixed, forces_mixed = report_system_state(mixed_system, positions)

    assert np.isfinite(energy_full)
    assert np.isfinite(energy_mixed)
    assert forces_full.shape == (6, 3)
    assert forces_mixed.shape == (6, 3)
    assert np.all(np.isfinite(forces_full))
    assert np.all(np.isfinite(forces_mixed))


def test_periodic_mixed_system_two_waters():
    """Verify periodic QM/MM system using pyscf.qmmm.pbc."""
    box_vectors = (
        unit.nanometer * mm.Vec3(2.0, 0.0, 0.0),
        unit.nanometer * mm.Vec3(0.0, 2.0, 0.0),
        unit.nanometer * mm.Vec3(0.0, 0.0, 2.0),
    )
    _, mixed_nonpbc, positions = create_pyscf_opemm_two_water_system()
    _, mixed_pbc, _ = create_pyscf_opemm_two_water_system(
        periodic_box_vectors=box_vectors
    )

    energy_nonpbc, forces_nonpbc = report_system_state(mixed_nonpbc, positions)
    energy_pbc, forces_pbc = report_system_state(mixed_pbc, positions)

    assert np.isfinite(energy_pbc)
    assert forces_pbc.shape == (6, 3)
    assert np.all(np.isfinite(forces_pbc))
    # Periodic boundary conditions include Ewald sum and periodic images,
    # so energy and forces must differ from non-periodic QM/MM.
    assert abs(energy_pbc - energy_nonpbc) > 0.5
    assert np.max(np.abs(forces_pbc - forces_nonpbc)) > 5.0


def test_periodic_mixed_system_parameters():
    """Verify specifying custom rcut_ewald, rcut_hcore, and mm_radii."""
    box_vectors = (
        unit.nanometer * mm.Vec3(2.0, 0.0, 0.0),
        unit.nanometer * mm.Vec3(0.0, 2.0, 0.0),
        unit.nanometer * mm.Vec3(0.0, 0.0, 2.0),
    )
    _, mixed_pbc, positions = create_pyscf_opemm_two_water_system(
        periodic_box_vectors=box_vectors,
        rcut_ewald=8.0,
        rcut_hcore=8.0,
        mm_radii=[1.0, 1.0, 1.0],
    )
    energy, forces = report_system_state(mixed_pbc, positions)
    assert np.isfinite(energy)
    assert forces.shape == (6, 3)


def test_periodic_mixed_system_rejects_non_orthogonal_box():
    """Verify non-orthogonal box vectors raise ValueError."""
    non_ortho_box = (
        unit.nanometer * mm.Vec3(2.0, 0.0, 0.0),
        unit.nanometer * mm.Vec3(0.5, 2.0, 0.0),
        unit.nanometer * mm.Vec3(0.0, 0.0, 2.0),
    )
    with pytest.raises(ValueError, match="orthogonal"):
        create_pyscf_opemm_two_water_system(periodic_box_vectors=non_ortho_box)


def test_periodic_mixed_system_finite_difference():
    """Verify analytical forces match numerical finite-difference gradient under PBC."""
    box_vectors = (
        unit.nanometer * mm.Vec3(2.0, 0.0, 0.0),
        unit.nanometer * mm.Vec3(0.0, 2.0, 0.0),
        unit.nanometer * mm.Vec3(0.0, 0.0, 2.0),
    )
    _, mixed_pbc, positions = create_pyscf_opemm_two_water_system(
        periodic_box_vectors=box_vectors
    )

    pos_nm = positions.value_in_unit(unit.nanometer).copy()
    energy_0, forces_ana = report_system_state(mixed_pbc, pos_nm * unit.nanometer)

    delta = 1e-4  # nm
    forces_num = np.zeros_like(pos_nm)

    integrator = mm.VerletIntegrator(0.5 * unit.femtosecond)
    context = mm.Context(mixed_pbc, integrator)
    try:
        for i in range(len(pos_nm)):
            for j in range(3):
                pos_p = pos_nm.copy()
                pos_m = pos_nm.copy()
                pos_p[i, j] += delta
                pos_m[i, j] -= delta

                context.setPositions(pos_p * unit.nanometer)
                ep = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

                context.setPositions(pos_m * unit.nanometer)
                em = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

                # Force is negative gradient: F = - dE / dx
                forces_num[i, j] = -(ep - em) / (2 * delta)
    finally:
        del context
        del integrator

    max_diff = np.max(np.abs(forces_ana - forces_num))
    print(f"Periodic QM/MM Max Force Diff (kJ/mol/nm): {max_diff}")
    assert max_diff < 0.5  # Excellent numerical agreement across 6 particles


def test_periodic_mixed_system_md():
    """Run short MD simulation for periodic QM/MM system."""
    box_vectors = (
        unit.nanometer * mm.Vec3(2.0, 0.0, 0.0),
        unit.nanometer * mm.Vec3(0.0, 2.0, 0.0),
        unit.nanometer * mm.Vec3(0.0, 0.0, 2.0),
    )
    _, mixed_system, positions = create_pyscf_opemm_two_water_system(
        periodic_box_vectors=box_vectors
    )
    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.5 * unit.femtosecond,
    )

    context = mm.Context(mixed_system, integrator)
    try:
        context.setPositions(positions)
        integrator.step(5)
        final_state = context.getState(getEnergy=True)
        energy = final_state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        assert np.isfinite(energy)
    finally:
        del context
        del integrator


if __name__ == "__main__":
    test_wrapper()
    test_solvation_configuration()
    test_mixed_system_api()
    test_mixed_system_rejects_solvation()
    test_two_waters_full_qm_and_mixed()
    test_periodic_mixed_system_two_waters()
    test_periodic_mixed_system_parameters()
    test_periodic_mixed_system_rejects_non_orthogonal_box()
    test_periodic_mixed_system_finite_difference()
    test_periodic_mixed_system_md()
    print("All tests passed successfully!")
