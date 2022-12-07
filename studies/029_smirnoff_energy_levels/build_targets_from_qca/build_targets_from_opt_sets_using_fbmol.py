import numpy as np
from openff.qcsubmit.results import OptimizationResultCollection
from collections import defaultdict
import pickle
from openeye import oechem
from simtk import unit
from pathlib import Path
from forcebalance.molecule import Molecule as FBMolecule
import os

# # Using Sage training set (OptimizationResultCollection)
# sage_training_set = OptimizationResultCollection.parse_file('1-2-0-opt-set-v3.json')
#
# # Downloading the records for all the QCA IDs in the json file above
# records_and_molecules = sage_training_set.to_records()
#
# # Writing to a pickle file
# with open('filename.pickle', 'wb') as handle:
#     pickle.dump(records_and_molecules, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('filename.pickle', 'rb') as handle:
    records_and_molecules = pickle.load(handle)

grouped_molecules = defaultdict(list)

# Group conformers together by SMILES
for record, molecule in records_and_molecules:
    molecule = molecule.canonical_order_atoms()
    smiles = molecule.to_smiles(isomeric=False, explicit_hydrogens=True)
    grouped_molecules[smiles].append((record, molecule))

# Create targets directory and write the input files
Path("./targets").mkdir(parents=True, exist_ok=True)
targets_input = open('./targets/energy_levels.in', 'w')
targets_excluded = open('./targets/targets_with_single_conformer_excluded.txt', 'w')

# Iterate over the molecules and the grouped together conformers
i = 0
for key, value in grouped_molecules.items():
    record, molecule = list(value)[0]
    # Check whether there are at least two conformers to take energy difference
    if len(value) < 2:
        print("Single molecule, need at least two conformers for energy levels target, ignoring this record", key)
        targets_excluded.write(f"QCA Record ID:{record.id}, SMILES: {molecule.to_smiles(mapped=True)}\n")
        continue
    i = i + 1

    target_dir = './targets/EL_QCA-' + str(i) + '-' + molecule.hill_formula
    name = os.path.basename(target_dir)
    # Write targets to a file that can be used in optimize_debug.in
    targets_input.write(f"$target \n"
                        f"  name {name}\n"
                        f"  type EnergyLevels_SMIRNOFF\n"
                        f"  mol2 mol-{i}.sdf\n"
                        f"  pdb mol-{i}.pdb\n"
                        f"  coords mol-{i}.xyz\n"
                        f"  writelevel 2\n"
                        f"  attenuate 1\n"
                        f"  restrain_k 1.0\n"
                        f"  energy_denom 1.0\n"
                        f"  energy_upper 5.0\n"
                        f"  e_width 0.5\n"
                        f"  weight 10.0\n"
                        f"  openmm_platform Reference\n"
                        f"$end\n"
                        f"\n")

    # Create target directory for this molecule EL_QCA-ID-hill_formula

    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # Extract energies from the optimization records
    energies = []
    for item in value:
        energies.append(item[0].get_final_energy())
    energies = np.array(energies)

    xyzs = []
    record_ids = []
    for j, index in enumerate(np.argsort(energies)):
        record, molecule = list(value)[index]
        xyzs.append(molecule.conformers[0].value_in_unit(unit.angstrom))
        record_ids.append(record.id)

    # Create a FB molecule object from the QCData molecule.
    fb_molecule = FBMolecule()
    fb_molecule.Data = {
        "resname": ["UNK"] * molecule.n_atoms,
        "resid": [0] * molecule.n_atoms,
        "elem": [atom.element.symbol for atom in molecule.atoms],
        "bonds": [
            (bond.atom1_index, bond.atom2_index) for bond in molecule.bonds
        ],
        "name": ', '.join(record_ids),
        "xyzs": xyzs,
        "qm_energies": list(np.sort(energies)),
        "comms": [f'QCA ID: {id}, Energy = {ener} hartree'for id, ener in zip(record_ids, np.sort(energies))]
    }
    coord_file_name = f'{target_dir}/mol-{str(i)}'
    # Write the data
    fb_molecule.write(target_dir+'/qdata.txt')
    fb_molecule.write(coord_file_name + '.xyz')

    # Write XYZ file in sorted energies order and sdf, pdb files of a single conformer for FB topologies
    sdf_file_output = f'{target_dir}/mol-{str(i)}.sdf'
    pdb_file_output = f'{target_dir}/mol-{str(i)}.pdb'

    # Write optgeo_options file within the energy level target directory
    fname = open(target_dir + '/optgeo_options.txt', 'w')

    fname.write("$global \n"
                "  bond_denom 0.01\n"
                "  angle_denom 3\n"
                "  dihedral_denom 10\n"
                "  improper_denom 10\n"
                "$end \n")
    fname.write(f"$system \n"
                f"  name {name}\n"
                f"  geometry mol-{i}.pdb\n"
                f"  topology mol-{i}.xyz\n"
                f"  mol2 mol-{i}.sdf\n"
                f"$end\n")


    molecule._properties["SMILES"] = molecule.to_smiles(mapped=True)
    molecule.to_file(sdf_file_output, file_format='sdf')
    fb_molecule.Data["xyzs"] = [fb_molecule.Data["xyzs"][0]]
    del fb_molecule.Data["qm_energies"]
    del fb_molecule.Data["comms"]
    fb_molecule.write(pdb_file_output)

targets_input.close()
