import numpy as np
from numpy import ndarray
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Mol, PeriodicTable
from math import pi


class Smiles2Descriptors:
    """
    Класс для вычисления дескрипторов молекул на основе SMILES-строки
    """

    def __init__(self, smiles: str) -> None:
        self.smiles: str = smiles
        self.molecule: Mol = Chem.MolFromSmiles(smiles)

        if self.molecule is None:
            raise ValueError("Неверный SMILES")

        #  молекула с водородом
        self.molecule_with_h: Mol = Chem.AddHs(self.molecule)

        # основные дескрипторы
        self.logp: float = Descriptors.MolLogP(self.molecule)
        self.tpsa: float = rdMolDescriptors.CalcTPSA(self.molecule)
        self.molecular_weight: float = Descriptors.MolWt(self.molecule)

        # характеристики связей
        self.total_bonds: int = self.molecule.GetNumBonds()
        self.rotatable_bonds: int = rdMolDescriptors.CalcNumRotatableBonds(
            self.molecule_with_h
            )
        self.non_rotatable_bonds: int = self.total_bonds - self.rotatable_bonds
        self.fraction_non_rotatable_bunds: float = (self.non_rotatable_bonds
                                                    / self.total_bonds)

        # объем по Ван-дер-Ваальсу
        self.vdw_volume: float = self._calculate_vdw_volume(
            self.molecule_with_h
            )

        self.num_atoms: int = self._find_atomic_number(
            self.molecule_with_h
        )

    @staticmethod
    def _calculate_vdw_volume(mol: Mol) -> float:
        volume: float = 0.0
        periodic_table: PeriodicTable = Chem.GetPeriodicTable()

        for atom in mol.GetAtoms():
            atomic_number: int = atom.GetAtomicNum()
            radius: float = periodic_table.GetRvdw(atomic_number)
            volume += (4 / 3) * pi * (radius ** 3)

        return volume

    @staticmethod
    def _find_atomic_number(mol: Mol) -> int:
        if mol is not None:
            num_atoms = mol.GetNumAtoms()

        return num_atoms

    def as_vector(self) -> ndarray:
        return np.array([
            self.logp,
            self.tpsa,
            self.molecular_weight,
            self.vdw_volume,
            self.fraction_non_rotatable_bunds,
            self.num_atoms
        ])

    def __repr__(self) -> str:
        return (
            f"SMILES: {self.smiles}\n"
            f"Descriptors:\n"
            f"  LogP: {self.logp:.2f}\n"
            f"  TPSA: {self.tpsa:.2f}\n"
            f"  Molecular Weight: {self.molecular_weight:.2f}\n"
            f"  Total Bonds: {self.total_bonds}\n"
            f"  Rotatable Bonds: {self.rotatable_bonds}\n"
            f"  Non-Rotatable Bonds: {self.non_rotatable_bonds}\n"
            f"  Van der Waals Volume: {self.vdw_volume:.2f}\n"
            f"  Fraction of non-rotatable bounds: {self.fraction_non_rotatable_bunds:.2f}\n"
            f"  Num atoms: {self.num_atoms}"
        )
