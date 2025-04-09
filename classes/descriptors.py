import numpy as np
from numpy import ndarray
from rdkit import Chem
from rdkit.Chem import (
    Descriptors, rdMolDescriptors,
    Mol, PeriodicTable,
    AllChem, rdPartialCharges
    )
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

        # Молекула с водородом
        self.mol_with_h: Mol = Chem.AddHs(self.molecule)

        # Основные дескрипторы
        # липофильность
        self.logp: float = Descriptors.MolLogP(self.molecule)
        # площадь поверхности полярных участков
        self.tpsa: float = rdMolDescriptors.CalcTPSA(self.molecule)
        # молярная масса
        self.molwt: float = Descriptors.MolWt(self.molecule)

        # Характеристики связей
        self.num_bounds: int = self.molecule.GetNumBonds()
        self.num_rotatable_bounds: int = rdMolDescriptors.CalcNumRotatableBonds(
            self.mol_with_h
        )
        self.num_non_rotatable_bounds: int = self.num_bounds - self.num_rotatable_bounds
        # доля не вращающихся связей
        self.fraction_non_rotatable_bounds: float = (
            self.num_non_rotatable_bounds / self.num_bounds
        )

        # Объем по Ван-дер-Ваальсу
        self.vdw_volume: float = self._calculate_vdw_volume(self.mol_with_h)

        self.num_atoms: int = self._find_atomic_number(self.mol_with_h)

        # Степень разветвленности (алгоритм Тараса Бондаренко)
        self.degree_of_branching: int = (
            self.smiles.count('(') / self.molecule.GetNumAtoms())

        # Дипольный момент
        self.dipole_moment: float = self.calculate_dipole()

        # fingerprints
        # self.fp_morgan = AllChem.GetMorganFingerprintAsBitVect(
        #     self.molecule, radius=2, nBits=1024
        # )
        # self.fp_maccs = Chem.MACCSkeys.GenMACCSKeys(self.molecule)
        # self.fp_topological = Chem.RDKFingerprint(self.molecule)

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

    def calculate_dipole(self) -> float:
        try:
            mol = Chem.MolFromSmiles(self.smiles)
            if mol is None:
                raise ValueError("Invalid SMILES")
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
            rdPartialCharges.ComputeGasteigerCharges(mol)
            conf = mol.GetConformer()
            dipole = np.zeros(3)
            for atom in mol.GetAtoms():
                charge = atom.GetDoubleProp("_GasteigerCharge") * 1.602176634e-19  # e -> C
                pos = conf.GetAtomPosition(atom.GetIdx())
                pos_m = np.array([pos.x, pos.y, pos.z]) * 1e-10  # A -> m
                dipole += charge * pos_m
            magnitude = np.linalg.norm(dipole)
            return magnitude / 3.33564e-30  # Перевод в Debye
        except Exception as e:
            print(f"Error calculating dipole moment: {str(e)}")
            return None

    def as_vector(self) -> ndarray:
        return np.array([
            self.logp,
            self.tpsa,
            self.molwt,
            self.vdw_volume,
            self.fraction_non_rotatable_bounds,
            self.num_atoms,
            self.degree_of_branching,
            self.dipole_moment
        ])

    def __repr__(self) -> str:
        return (
            f"SMILES: {self.smiles}\n"
            f"Descriptors:\n"
            f"  LogP: {self.logp:.2f}\n"
            f"  TPSA: {self.tpsa:.2f}\n"
            f"  Molecular Weight: {self.molwt:.2f}\n"
            f"  Total Bonds: {self.num_bounds}\n"
            f"  Rotatable Bonds: {self.num_rotatable_bounds}\n"
            f"  Non-Rotatable Bonds: {self.num_non_rotatable_bounds}\n"
            f"  Van der Waals Volume: {self.vdw_volume:.2f}\n"
            f"  Fraction of non-rotatable bounds: {
                self.fraction_non_rotatable_bounds:.2f}\n"
            f"  Num atoms: {self.num_atoms}\n"
            f"  Degree of branching: {self.degree_of_branching}\n"
            f"  Dipole Moment: {self.dipole_moment:.2f} D\n"
            # f"  Morgan fingerprint: {self.fp_morgan}\n"
            # f"  MACCS fingerprint: {self.fp_maccs}\n"
            # f"  Topological fingerprint: {self.fp_topological}\n"
        )
