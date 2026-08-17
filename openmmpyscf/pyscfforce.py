from __future__ import annotations

from dataclasses import dataclass
import xml.etree.ElementTree as ET
from typing import Mapping
import warnings

import numpy as np
import openmm as mm
import openmm.app
from openmm import unit
from pyscf import dft, gto, lib, scf


# Ensure numpy 2.x compatibility for PySCF einsum
def _patch_pyscf_einsum():
    try:
        if hasattr(lib, "numpy_helper") and hasattr(lib.numpy_helper, "einsum"):
            def _safe_einsum(subscripts, *tensors, **kwargs):
                contract = kwargs.pop('_contract', lib.numpy_helper._contract)
                subscripts = subscripts.replace(' ', '')
                if len(tensors) <= 1 or '...' in subscripts:
                    return np.einsum(subscripts, *tensors, **kwargs)
                elif len(tensors) <= 2:
                    return lib.numpy_helper._contract(subscripts, *tensors, **kwargs)
                else:
                    optimize = kwargs.pop('optimize', True)
                    tensors = list(tensors)
                    contraction_list = np.einsum_path(subscripts, *tensors, optimize=optimize, einsum_call=True)[1]
                    for contraction in contraction_list:
                        if len(contraction) == 3:
                            inds, einsum_str, remaining = contraction
                        else:
                            inds, idx_rm, einsum_str, remaining = contraction[:4]
                        tmp_operands = [tensors.pop(x) for x in inds]
                        if len(tmp_operands) > 2:
                            out = np.einsum(einsum_str, *tmp_operands)
                        else:
                            out = contract(einsum_str, *tmp_operands)
                        tensors.append(out)
                    return out

            lib.einsum = _safe_einsum
            lib.numpy_helper.einsum = _safe_einsum
    except Exception:
        pass


_patch_pyscf_einsum()

HARTREE_TO_KJMOL = 2625.499639
ANGSTROM_PER_NANOMETER = 10.0
BOHR_PER_ANGSTROM = 1.8897261254578281


@dataclass(frozen=True)
class PySCFResult:
    energy: unit.Quantity
    forces: unit.Quantity


class PySCFPythonForce:
    """OpenMM callback wrapper that evaluates PySCF energies and gradients."""

    def __init__(
        self,
        symbols,
        method="b3lyp",
        basis="3-21g",
        charge=0,
        multiplicity=1,
        memory="1 GB",
        num_threads=None,
        pyscf_options: Mapping[str, object] | None = None,
        solvation_model: str | None = None,
        solvation_options: Mapping[str, object] | None = None,
        density_fit: bool = True,
        use_gpu=False,
        quiet=True,
    ):
        self.symbols = [str(sym).strip() for sym in symbols]
        self.method = str(method)
        self.basis = str(basis)
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        self.memory_mb = self._parse_memory_mb(memory)
        self.num_threads = None if num_threads is None else int(num_threads)
        self.pyscf_options = dict(pyscf_options) if pyscf_options else {}
        self.solvation_model = self._normalize_solvation_model(solvation_model)
        self.solvation_options = dict(solvation_options) if solvation_options else {}
        self.density_fit = bool(density_fit)
        self.use_gpu = bool(use_gpu)
        self.quiet = bool(quiet)
        self._cached_qm_mol = None
        self._cached_qm_model_by_df = {}
        self._runtime_density_fit = bool(density_fit)
        self._df_fallback_warned = False

        if not self.symbols:
            raise ValueError("symbols must be a non-empty sequence")
        if any(not sym for sym in self.symbols):
            raise ValueError("symbols contains an empty atomic symbol")
        if self.multiplicity < 1:
            raise ValueError("multiplicity must be at least 1")

        if self.num_threads is not None:
            lib.num_threads(self.num_threads)

    @staticmethod
    def _parse_memory_mb(memory) -> int:
        if isinstance(memory, (int, float)):
            return int(memory)

        text = str(memory).strip().lower()
        parts = text.split()
        if len(parts) == 1:
            return int(float(parts[0]))

        value = float(parts[0])
        unit_name = parts[1]
        factors = {
            "mb": 1,
            "mib": 1,
            "gb": 1024,
            "gib": 1024,
            "kb": 1 / 1024,
            "kib": 1 / 1024,
        }
        if unit_name not in factors:
            raise ValueError(f"Unsupported memory unit '{unit_name}'")
        return int(value * factors[unit_name])

    def _positions_nm_from_state(self, state) -> np.ndarray:
        positions = state.getPositions(asNumpy=True)
        if positions is None:
            raise ValueError("OpenMM state does not provide positions")

        coords_nm = np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)
        if coords_nm.ndim != 2 or coords_nm.shape[1] != 3:
            raise ValueError(f"Expected coordinates shape (N, 3), got {coords_nm.shape}")
        if coords_nm.shape[0] != len(self.symbols):
            raise ValueError(
                f"Atom count mismatch: {len(self.symbols)} symbols but {coords_nm.shape[0]} positions"
            )
        return coords_nm

    @staticmethod
    def _normalize_solvation_model(solvation_model: str | None) -> str | None:
        if solvation_model is None:
            return None

        model = str(solvation_model).strip()
        if not model or model.lower() in {"none", "false", "no", "off"}:
            return None
        return model

    @staticmethod
    def _normalize_periodic_box_vectors(periodic_box_vectors) -> np.ndarray | None:
        if periodic_box_vectors is None:
            return None

        vectors = []
        for vector in periodic_box_vectors:
            if hasattr(vector, "value_in_unit"):
                vectors.append(np.asarray(vector.value_in_unit(unit.angstrom), dtype=float))
            else:
                vectors.append(np.asarray(vector, dtype=float))

        array = np.asarray(vectors, dtype=float)
        if array.shape != (3, 3):
            raise ValueError(
                f"Periodic box vectors must have shape (3, 3), got {array.shape}"
            )
        return array

    def _apply_solvation_model(self, mf):
        if self.solvation_model is None:
            return mf

        try:
            from pyscf import solvent
        except Exception as exc:
            raise RuntimeError(
                f"PySCF solvation model '{self.solvation_model}' was requested, but the solvent module is unavailable"
            ) from exc

        model_key = self.solvation_model.replace("-", "").replace("_", "").lower()
        factory_name_map = {
            "ddcosmo": "ddCOSMO",
            "pcm": "PCM",
            "smd": "SMD",
        }
        factory_name = factory_name_map.get(model_key)
        if factory_name is None:
            candidate_names = [
                self.solvation_model,
                self.solvation_model.lower(),
                self.solvation_model.upper(),
                self.solvation_model.title(),
            ]
            for candidate_name in candidate_names:
                if hasattr(solvent, candidate_name):
                    factory_name = candidate_name
                    break

        if factory_name is None or not hasattr(solvent, factory_name):
            available_models = [name for name in dir(solvent) if not name.startswith("_")]
            raise ValueError(
                f"Unsupported PySCF solvation model '{self.solvation_model}'. "
                f"Available solvent helpers include: {', '.join(sorted(available_models))}"
            )

        solvent_factory = getattr(solvent, factory_name)
        solvation_mf = solvent_factory(mf)

        for key, value in self.solvation_options.items():
            setattr(solvation_mf, key, value)

        return solvation_mf

    def _build_qm_system(self, positions_ang: np.ndarray):
        atom = [(symbol, tuple(xyz)) for symbol, xyz in zip(self.symbols, positions_ang)]
        spin = self.multiplicity - 1
        verbose = 0 if self.quiet else 4
        if self._cached_qm_mol is None:
            self._cached_qm_mol = gto.M(
                atom=atom,
                basis=self.basis,
                charge=self.charge,
                spin=spin,
                unit="Angstrom",
                verbose=verbose,
                max_memory=self.memory_mb,
            )
        else:
            self._cached_qm_mol.set_geom_(atom, unit="Angstrom")
        return self._cached_qm_mol

    def _add_mm_charges_to_qm_model(self, qm_model, positions_ang: np.ndarray):
        return qm_model

    def _is_qm_model_cacheable(self) -> bool:
        return True

    def _apply_density_fit(self, qm_model, is_dft: bool = False):
        if not self._runtime_density_fit:
            return qm_model

        qm_model = qm_model.density_fit()
        # For DFT, prefer a compact J-fitting basis as recommended by PySCF docs.
        if is_dft and getattr(qm_model, "with_df", None) is not None:
            qm_model.with_df.auxbasis = "def2-universal-jfit"
        return qm_model

    def _build_method(self, mol, positions_ang: np.ndarray | None = None):
        if self._is_qm_model_cacheable():
            cache_key = bool(self._runtime_density_fit)
            cached_qm_model = self._cached_qm_model_by_df.get(cache_key)
            if cached_qm_model is None:
                cached_qm_model = self._build_method_uncached(mol, positions_ang)
                self._cached_qm_model_by_df[cache_key] = cached_qm_model
            else:
                cached_qm_model.reset(mol)
            return cached_qm_model

        return self._build_method_uncached(mol, positions_ang)

    def _build_method_uncached(self, mol, positions_ang: np.ndarray | None = None):
        method = self.method.lower()
        spin = self.multiplicity - 1

        if method in ("hf", "rhf"):
            mf = scf.RHF(mol) if spin == 0 else scf.UHF(mol)
            is_dft = False
        elif method == "uhf":
            mf = scf.UHF(mol)
            is_dft = False
        elif method == "rohf":
            mf = scf.ROHF(mol)
            is_dft = False
        else:
            mf = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
            mf.xc = self.method
            is_dft = True

        for key, value in self.pyscf_options.items():
            setattr(mf, key, value)

        mf = self._apply_density_fit(mf, is_dft=is_dft)

        if self.use_gpu:
            try:
                import gpu4pyscf  # noqa: F401

                if hasattr(mf, "to_gpu"):
                    mf = mf.to_gpu()
                else:
                    warnings.warn(
                        "GPU requested (use_gpu=True) but this method does not support to_gpu(); using CPU",
                        RuntimeWarning,
                    )
            except Exception as exc:
                warnings.warn(
                    f"GPU requested (use_gpu=True) but GPU4PySCF is unavailable ({exc}); using CPU",
                    RuntimeWarning,
                )

        mf = self._add_mm_charges_to_qm_model(mf, positions_ang)
        mf = self._apply_solvation_model(mf)

        return mf

    def _evaluate_energy_hartree(self, positions_nm: np.ndarray) -> float:
        positions_ang = positions_nm * ANGSTROM_PER_NANOMETER
        qm_system = self._build_qm_system(positions_ang)
        qm_model = self._build_method(qm_system, positions_ang)

        try:
            return float(qm_model.kernel())
        except Exception as exc:
            raise RuntimeError(
                f"PySCF energy evaluation failed for method='{self.method}' basis='{self.basis}'"
            ) from exc

    def _run_nonperiodic_qm(self, positions_ang: np.ndarray):
        molecule = self._build_qm_system(positions_ang)
        qm_model = self._build_method(molecule, positions_ang)
        energy_hartree = float(qm_model.kernel())
        gradient_hartree_per_bohr = np.asarray(qm_model.nuc_grad_method().kernel(), dtype=float)
        return energy_hartree, gradient_hartree_per_bohr

    def _evaluate_nonperiodic(self, positions_nm: np.ndarray) -> PySCFResult:
        positions_ang = positions_nm * ANGSTROM_PER_NANOMETER

        try:
            energy_hartree, gradient_hartree_per_bohr = self._run_nonperiodic_qm(positions_ang)
        except Exception as exc:
            if not self._runtime_density_fit:
                raise RuntimeError(
                    f"PySCF gradient evaluation failed for method='{self.method}' basis='{self.basis}'"
                ) from exc

            if not self._df_fallback_warned:
                warnings.warn(
                    "Density-fitted PySCF gradients failed; disabling density fitting for subsequent evaluations",
                    RuntimeWarning,
                )
                self._df_fallback_warned = True
            try:
                self._runtime_density_fit = False
                energy_hartree, gradient_hartree_per_bohr = self._run_nonperiodic_qm(positions_ang)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"PySCF gradient evaluation failed for method='{self.method}' basis='{self.basis}'"
                ) from fallback_exc

        if gradient_hartree_per_bohr.shape != positions_nm.shape:
            raise RuntimeError(
                f"Gradient shape mismatch: expected {positions_nm.shape}, got {gradient_hartree_per_bohr.shape}"
            )

        energy_kj_mol = energy_hartree * HARTREE_TO_KJMOL
        force_kj_mol_nm = (
            -gradient_hartree_per_bohr
            * HARTREE_TO_KJMOL
            * BOHR_PER_ANGSTROM
            * ANGSTROM_PER_NANOMETER
        )

        return PySCFResult(
            energy=energy_kj_mol * unit.kilojoules_per_mole,
            forces=force_kj_mol_nm * unit.kilojoules_per_mole / unit.nanometer,
        )

    def _evaluate(self, positions_nm: np.ndarray) -> PySCFResult:
        return self._evaluate_nonperiodic(positions_nm)

    def compute(self, state):
        """OpenMM callback returning (energy, forces) quantities."""
        positions_nm = self._positions_nm_from_state(state)
        result = self._evaluate(positions_nm)
        return result.energy, result.forces

    def energy_and_forces(self, positions):
        """Direct helper for testing with explicit coordinates in nanometers."""
        if hasattr(positions, "value_in_unit"):
            coords_nm = np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)
        else:
            coords_nm = np.asarray(positions, dtype=float)
        if coords_nm.shape[0] != len(self.symbols):
            raise ValueError(
                f"Atom count mismatch: {len(self.symbols)} symbols but {coords_nm.shape[0]} coordinates"
            )
        return self._evaluate(coords_nm)


class _ElectronicEmbeddingPySCFPythonForce(PySCFPythonForce):
    """PySCF callback for electrostatic QM/MM embedding."""

    def __init__(
        self,
        base_force: PySCFPythonForce,
        qm_symbols: list[str],
        qm_atoms: list[int],
        mm_atoms: list[int],
        mm_charges: list[float],
        periodic_box_vectors=None,
        rcut_ewald: float | None = None,
        rcut_hcore: float | None = None,
        mm_radii: float | list[float] | np.ndarray | None = None,
    ):
        if base_force.solvation_model is not None:
            raise ValueError(
                "QM/MM electronic embedding does not support PySCF solvation models in this wrapper"
            )
        self.qm_atoms = tuple(int(atom) for atom in qm_atoms)
        self.mm_atoms = tuple(int(atom) for atom in mm_atoms)
        self.total_atom_count = len(self.qm_atoms) + len(self.mm_atoms)
        self.mm_charges = np.asarray(mm_charges, dtype=float)
        self.periodic_box_vectors_ang = self._normalize_periodic_box_vectors(periodic_box_vectors)

        if self.periodic_box_vectors_ang is not None:
            # Check orthogonality
            if not np.allclose(
                self.periodic_box_vectors_ang,
                np.diag(np.diag(self.periodic_box_vectors_ang)),
                atol=1e-6,
            ):
                raise ValueError("pyscf.qmmm.pbc requires orthogonal (diagonal) periodic box vectors")

            diag_a = np.diag(self.periodic_box_vectors_ang)
            if rcut_ewald is None:
                self.rcut_ewald = float(min(diag_a) * 0.5)
            else:
                self.rcut_ewald = float(rcut_ewald)

            if rcut_hcore is None:
                self.rcut_hcore = float(min(diag_a) * 0.5)
            else:
                self.rcut_hcore = float(rcut_hcore)
        else:
            self.rcut_ewald = float(rcut_ewald) if rcut_ewald is not None else None
            self.rcut_hcore = float(rcut_hcore) if rcut_hcore is not None else None

        if mm_radii is None:
            self.mm_radii = np.ones(len(self.mm_atoms), dtype=float) if len(self.mm_atoms) > 0 else None
        elif isinstance(mm_radii, (int, float)):
            self.mm_radii = np.full(len(self.mm_atoms), float(mm_radii), dtype=float)
        else:
            radii_arr = np.asarray(mm_radii, dtype=float).ravel()
            if len(radii_arr) != len(self.mm_atoms):
                raise ValueError(
                    f"mm_radii length mismatch: expected {len(self.mm_atoms)} but got {len(radii_arr)}"
                )
            self.mm_radii = radii_arr

        super().__init__(
            symbols=qm_symbols,
            method=base_force.method,
            basis=base_force.basis,
            charge=base_force.charge,
            multiplicity=base_force.multiplicity,
            memory=base_force.memory_mb,
            num_threads=base_force.num_threads,
            pyscf_options=base_force.pyscf_options,
            density_fit=base_force.density_fit,
            use_gpu=base_force.use_gpu,
            quiet=base_force.quiet,
        )

    def _positions_nm_from_state(self, state) -> np.ndarray:
        positions = state.getPositions(asNumpy=True)
        if positions is None:
            raise ValueError("OpenMM state does not provide positions")

        coords_nm = np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)
        if coords_nm.ndim != 2 or coords_nm.shape[1] != 3:
            raise ValueError(f"Expected coordinates shape (N, 3), got {coords_nm.shape}")
        if coords_nm.shape[0] != self.total_atom_count:
            raise ValueError(
                f"Atom count mismatch: expected {self.total_atom_count} positions but got {coords_nm.shape[0]}"
            )
        return coords_nm

    def energy_and_forces(self, positions):
        if hasattr(positions, "value_in_unit"):
            coords_nm = np.asarray(positions.value_in_unit(unit.nanometer), dtype=float)
        else:
            coords_nm = np.asarray(positions, dtype=float)
        if coords_nm.shape[0] != self.total_atom_count:
            raise ValueError(
                f"Atom count mismatch: expected {self.total_atom_count} coordinates but got {coords_nm.shape[0]}"
            )
        return self._evaluate(coords_nm)

    def _add_mm_charges_to_qm_model(self, qm_model, positions_ang: np.ndarray):
        if len(self.mm_atoms) == 0:
            return qm_model

        mm_positions_ang = positions_ang[list(self.mm_atoms)]

        if self.periodic_box_vectors_ang is None:
            from pyscf import qmmm
            qm_model = qmmm.mm_charge(qm_model, mm_positions_ang, self.mm_charges)
        else:
            from pyscf.qmmm.pbc import itrf as pbc_itrf
            qm_model = pbc_itrf.add_mm_charges(
                qm_model,
                mm_positions_ang,
                self.periodic_box_vectors_ang,
                self.mm_charges,
                radii=self.mm_radii,
                rcut_ewald=self.rcut_ewald,
                rcut_hcore=self.rcut_hcore,
                unit="Angstrom",
            )

        return qm_model

    def _is_qm_model_cacheable(self) -> bool:
        # Step-dependent MM coordinates require rebuilding the decorated SCF object.
        return False

    def _run_embedded_qm(self, positions_ang: np.ndarray):
        qm_positions_ang = positions_ang[list(self.qm_atoms)]
        mm_positions_ang = positions_ang[list(self.mm_atoms)]

        molecule = self._build_qm_system(qm_positions_ang)
        qm_model = self._build_method(molecule, positions_ang)

        energy_hartree = float(qm_model.kernel())
        gradient = qm_model.nuc_grad_method()

        if self.periodic_box_vectors_ang is not None:
            if getattr(qm_model, "with_df", None) is not None:
                gradient.auxbasis_response = True
            qm_gradient_hartree_per_bohr = np.asarray(gradient.kernel(), dtype=float)
            if len(self.mm_atoms) > 0:
                dm = qm_model.make_rdm1()
                mm_gradient_hartree_per_bohr = np.asarray(
                    gradient.grad_nuc_mm() + gradient.grad_hcore_mm(dm) + gradient.de_ewald_mm,
                    dtype=float,
                )
            else:
                mm_gradient_hartree_per_bohr = np.zeros((0, 3), dtype=float)
        else:
            qm_gradient_hartree_per_bohr = np.asarray(gradient.kernel(), dtype=float)
            if len(self.mm_atoms) > 0:
                mm_gradient_hartree_per_bohr = np.asarray(
                    gradient.grad_hcore_mm(qm_model.make_rdm1()) + gradient.grad_nuc_mm(),
                    dtype=float,
                )
            else:
                mm_gradient_hartree_per_bohr = np.zeros((0, 3), dtype=float)

        return energy_hartree, qm_positions_ang, mm_positions_ang, qm_gradient_hartree_per_bohr, mm_gradient_hartree_per_bohr

    def _evaluate(self, positions_nm: np.ndarray) -> PySCFResult:
        positions_ang = positions_nm * ANGSTROM_PER_NANOMETER
        try:
            (
                energy_hartree,
                qm_positions_ang,
                mm_positions_ang,
                qm_gradient_hartree_per_bohr,
                mm_gradient_hartree_per_bohr,
            ) = self._run_embedded_qm(positions_ang)
        except Exception as exc:
            if not self._runtime_density_fit:
                raise RuntimeError(
                    f"PySCF QM/MM evaluation failed for method='{self.method}' basis='{self.basis}'"
                ) from exc

            if not self._df_fallback_warned:
                warnings.warn(
                    "Density-fitted PySCF QM/MM gradients failed; disabling density fitting for subsequent evaluations",
                    RuntimeWarning,
                )
                self._df_fallback_warned = True
            try:
                self._runtime_density_fit = False
                (
                    energy_hartree,
                    qm_positions_ang,
                    mm_positions_ang,
                    qm_gradient_hartree_per_bohr,
                    mm_gradient_hartree_per_bohr,
                ) = self._run_embedded_qm(positions_ang)
            except Exception as fallback_exc:
                raise RuntimeError(
                    f"PySCF QM/MM evaluation failed for method='{self.method}' basis='{self.basis}'"
                ) from fallback_exc

        if qm_gradient_hartree_per_bohr.shape != qm_positions_ang.shape:
            raise RuntimeError(
                f"QM gradient shape mismatch: expected {qm_positions_ang.shape}, got {qm_gradient_hartree_per_bohr.shape}"
            )
        if mm_gradient_hartree_per_bohr.shape != mm_positions_ang.shape:
            raise RuntimeError(
                f"MM gradient shape mismatch: expected {mm_positions_ang.shape}, got {mm_gradient_hartree_per_bohr.shape}"
            )

        full_gradient_hartree_per_bohr = np.zeros_like(positions_ang, dtype=float)
        full_gradient_hartree_per_bohr[list(self.qm_atoms)] = qm_gradient_hartree_per_bohr
        if len(self.mm_atoms) > 0:
            full_gradient_hartree_per_bohr[list(self.mm_atoms)] = mm_gradient_hartree_per_bohr

        energy_kj_mol = energy_hartree * HARTREE_TO_KJMOL
        force_kj_mol_nm = (
            -full_gradient_hartree_per_bohr
            * HARTREE_TO_KJMOL
            * BOHR_PER_ANGSTROM
            * ANGSTROM_PER_NANOMETER
        )

        return PySCFResult(
            energy=energy_kj_mol * unit.kilojoules_per_mole,
            forces=force_kj_mol_nm * unit.kilojoules_per_mole / unit.nanometer,
        )


def make_openmm_python_force(pyscf_force: PySCFPythonForce) -> mm.Force:
    """Build an OpenMM PythonForce from a PySCFPythonForce instance."""
    return mm.PythonForce(pyscf_force.compute)


class PySCFPotential:
    """Potential object that builds OpenMM Systems from a Topology.

    This follows the openmm-ml style:

    >>> potential = PySCFPotential(method="b3lyp", basis="6-31g*")
    >>> system = potential.createSystem(topology)
    """

    def __init__(
        self,
        method="b3lyp",
        basis="3-21g",
        charge=0,
        multiplicity=1,
        memory="1 GB",
        num_threads=None,
        pyscf_options: Mapping[str, object] | None = None,
        solvation_model: str | None = None,
        solvation_options: Mapping[str, object] | None = None,
        density_fit: bool = True,
        use_gpu=False,
        quiet=True,
        rcut_ewald: float | None = None,
        rcut_hcore: float | None = None,
        mm_radii: float | list[float] | np.ndarray | None = None,
    ):
        self.method = str(method)
        self.basis = str(basis)
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        self.memory = memory
        self.num_threads = num_threads
        self.pyscf_options = dict(pyscf_options) if pyscf_options else None
        self.solvation_model = solvation_model
        self.solvation_options = dict(solvation_options) if solvation_options else None
        self.density_fit = bool(density_fit)
        self.use_gpu = bool(use_gpu)
        self.quiet = bool(quiet)
        self.rcut_ewald = rcut_ewald
        self.rcut_hcore = rcut_hcore
        self.mm_radii = mm_radii

    @staticmethod
    def _periodic_box_vectors_from_topology(topology: openmm.app.Topology):
        return topology.getPeriodicBoxVectors()

    @staticmethod
    def _periodic_box_vectors_for_mixed_system(
        topology: openmm.app.Topology,
        system: mm.System,
        periodic_box_vectors=None,
    ):
        if periodic_box_vectors is not None:
            return periodic_box_vectors

        topology_vectors = topology.getPeriodicBoxVectors()
        if topology_vectors is not None:
            return topology_vectors

        if system.usesPeriodicBoundaryConditions():
            return system.getDefaultPeriodicBoxVectors()

        return None

    def _symbols_from_topology(self, topology: openmm.app.Topology) -> list[str]:
        symbols: list[str] = []
        for atom in topology.atoms():
            if atom.element is None:
                raise ValueError(
                    "Topology contains atoms without elements; cannot infer PySCF symbols"
                )
            symbols.append(atom.element.symbol)
        if not symbols:
            raise ValueError("Topology has no atoms")
        return symbols

    @staticmethod
    def _copy_system(system: mm.System) -> mm.System:
        return mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))

    @staticmethod
    def _get_nonbonded_force(system: mm.System) -> mm.NonbondedForce:
        nonbonded_forces = [force for force in system.getForces() if isinstance(force, mm.NonbondedForce)]
        if not nonbonded_forces:
            raise ValueError("QM/MM mixed systems require a NonbondedForce in the input system")
        if len(nonbonded_forces) > 1:
            raise NotImplementedError("Systems with multiple NonbondedForce objects are not supported")
        return nonbonded_forces[0]

    @staticmethod
    def _remove_bonded_terms(system: mm.System, atoms: list[int]) -> mm.System:
        atom_set = set(atoms)
        root = ET.fromstring(mm.XmlSerializer.serialize(system))

        def should_remove(term_atoms: list[int]) -> bool:
            return any(atom in atom_set for atom in term_atoms)

        for bonds in root.findall('./Forces/Force/Bonds'):
            for bond in list(bonds):
                bond_atoms = [int(bond.attrib['p1']), int(bond.attrib['p2'])]
                if should_remove(bond_atoms):
                    bonds.remove(bond)

        for angles in root.findall('./Forces/Force/Angles'):
            for angle in list(angles):
                angle_atoms = [int(angle.attrib['p1']), int(angle.attrib['p2']), int(angle.attrib['p3'])]
                if should_remove(angle_atoms):
                    angles.remove(angle)

        for torsions in root.findall('./Forces/Force/Torsions'):
            for torsion in list(torsions):
                labels = ('p1', 'p2', 'p3', 'p4') if 'p1' in torsion.attrib else ('a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3', 'b4')
                torsion_atoms = [int(torsion.attrib[label]) for label in labels]
                if should_remove(torsion_atoms):
                    torsions.remove(torsion)

        return mm.XmlSerializer.deserialize(ET.tostring(root, encoding='unicode'))

    @staticmethod
    def _apply_electrostatic_embedding(system: mm.System, qm_atoms: list[int]) -> None:
        force = PySCFPotential._get_nonbonded_force(system)
        qm_set = set(qm_atoms)
        exception_lookup = {
            tuple(sorted(force.getExceptionParameters(i)[:2])): i
            for i in range(force.getNumExceptions())
        }

        def pair_sigma_epsilon(atom1: int, atom2: int) -> tuple[float, float]:
            _, sigma1, epsilon1 = force.getParticleParameters(atom1)
            _, sigma2, epsilon2 = force.getParticleParameters(atom2)
            return 0.5 * (sigma1 + sigma2), (epsilon1 * epsilon2) ** 0.5

        for atom1 in range(force.getNumParticles()):
            for atom2 in range(atom1):
                if atom1 not in qm_set and atom2 not in qm_set:
                    continue

                sigma, epsilon = pair_sigma_epsilon(atom1, atom2)
                charge_prod = 0.0
                if atom1 in qm_set and atom2 in qm_set:
                    epsilon = 0.0

                key = (atom2, atom1)
                if key in exception_lookup:
                    force.setExceptionParameters(exception_lookup[key], atom1, atom2, charge_prod, sigma, epsilon)
                else:
                    force.addException(atom1, atom2, charge_prod, sigma, epsilon, True)

    def getSupportedEmbeddings(self) -> list[str]:
        return ["electronic"]

    def createMixedSystem(
        self,
        topology: openmm.app.Topology,
        system: mm.System,
        atoms: list[int],
        removeConstraints: bool = True,
        forceGroup: int = 0,
        interpolate: bool = False,
        embedding: str = "electronic",
        **args,
    ) -> mm.System:
        """Create a mixed QM/MM System using electronic embedding.

        The QM subsystem is evaluated with PySCF and is electrostatically
        embedded in the MM point charges of the remaining atoms.
        """

        if interpolate:
            raise NotImplementedError("Interpolation is not supported by PySCFPotential.createMixedSystem().")
        if embedding != "electronic":
            raise ValueError(f"Unsupported embedding '{embedding}'; only 'electronic' is available")
        if self.solvation_model is not None:
            raise ValueError(
                "PySCFPotential.createMixedSystem() does not support solvation_model; "
                "implicit solvation is incompatible with this QM/MM electronic-embedding implementation"
            )

        atom_list = [int(atom) for atom in atoms]
        if not atom_list:
            raise ValueError("atoms must be a non-empty sequence")
        if len(set(atom_list)) != len(atom_list):
            raise ValueError("atoms contains duplicate indices")

        if min(atom_list) < 0:
            raise ValueError("atoms contains a negative index")

        symbols = self._symbols_from_topology(topology)
        if max(atom_list) >= len(symbols):
            raise ValueError("atoms contains an index outside the topology")

        qm_set = set(atom_list)
        mm_atoms = [atom_index for atom_index in range(len(symbols)) if atom_index not in qm_set]
        qm_symbols = [symbols[atom_index] for atom_index in atom_list]
        periodic_box_vectors = self._periodic_box_vectors_for_mixed_system(
            topology,
            system,
            periodic_box_vectors=args.pop("periodicBoxVectors", args.pop("periodic_box_vectors", None)),
        )

        rcut_ewald = args.pop("rcut_ewald", args.pop("rcutEwald", self.rcut_ewald))
        rcut_hcore = args.pop("rcut_hcore", args.pop("rcutHcore", self.rcut_hcore))
        mm_radii = args.pop("mm_radii", args.pop("radii", args.pop("mmRadii", self.mm_radii)))

        mixed_system = self._remove_bonded_terms(self._copy_system(system), atom_list)
        self._apply_electrostatic_embedding(mixed_system, atom_list)
        if periodic_box_vectors is not None:
            mixed_system.setDefaultPeriodicBoxVectors(*periodic_box_vectors)

        if removeConstraints:
            atom_set = set(atom_list)
            constraints_to_remove: list[int] = []
            for constraint_index in range(mixed_system.getNumConstraints()):
                p1, p2, _ = mixed_system.getConstraintParameters(constraint_index)
                if p1 in atom_set and p2 in atom_set:
                    constraints_to_remove.append(constraint_index)
            for constraint_index in reversed(constraints_to_remove):
                mixed_system.removeConstraint(constraint_index)

        base_force = self.createCalculator(topology)
        qm_atom_charges = []
        nonbonded_force = self._get_nonbonded_force(system)
        for atom_index in mm_atoms:
            charge, _, _ = nonbonded_force.getParticleParameters(atom_index)
            qm_atom_charges.append(charge.value_in_unit(unit.elementary_charge))

        qm_force = _ElectronicEmbeddingPySCFPythonForce(
            base_force=base_force,
            qm_symbols=qm_symbols,
            qm_atoms=atom_list,
            mm_atoms=mm_atoms,
            mm_charges=qm_atom_charges,
            periodic_box_vectors=periodic_box_vectors,
            rcut_ewald=rcut_ewald,
            rcut_hcore=rcut_hcore,
            mm_radii=mm_radii,
        )
        qm_force_handle = make_openmm_python_force(qm_force)
        qm_force_handle.setForceGroup(forceGroup)
        mixed_system.addForce(qm_force_handle)

        return mixed_system

    def createCalculator(self, topology: openmm.app.Topology) -> PySCFPythonForce:
        """Create a PySCFPythonForce callback object from a Topology."""
        symbols = self._symbols_from_topology(topology)
        return PySCFPythonForce(
            symbols=symbols,
            method=self.method,
            basis=self.basis,
            charge=self.charge,
            multiplicity=self.multiplicity,
            memory=self.memory,
            num_threads=self.num_threads,
            pyscf_options=self.pyscf_options,
            solvation_model=self.solvation_model,
            solvation_options=self.solvation_options,
            density_fit=self.density_fit,
            use_gpu=self.use_gpu,
            quiet=self.quiet,
        )

    def createSystem(
        self, topology: openmm.app.Topology, removeCMMotion: bool = True
    ) -> mm.System:
        """Create an OpenMM System for this Topology using the PySCF potential."""
        system = mm.System()

        for atom in topology.atoms():
            if atom.element is None:
                system.addParticle(0)
            else:
                system.addParticle(atom.element.mass)

        pyscf_force = self.createCalculator(topology)
        system.addForce(make_openmm_python_force(pyscf_force))

        if removeCMMotion:
            system.addForce(mm.CMMotionRemover())

        return system
