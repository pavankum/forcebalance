import numpy as np
import qcportal
from openff.qcsubmit.results import OptimizationResultCollection
from collections import defaultdict
import pickle
from openeye import oechem
from openff.toolkit.topology import Molecule
from simtk import unit
from pathlib import Path
from forcebalance.molecule import Molecule as FBMolecule
import os
from forcebalance.nifty import au2kcal, fqcgmx, au2kj

client = qcportal.FractalClient.from_file()

# phalkethoh_set = OptimizationResultCollection.from_server(client=client,
#                                                           datasets=['OpenFF Sandbox CHO PhAlkEthOH v1.0'],
#                                                           spec_name='default')
# records_and_molecules = phalkethoh_set.to_records()
# with open('phalkethoh.pickle', 'wb') as handle:
#     pickle.dump(records_and_molecules, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
    # molecule = molecule.canonical_order_atoms()
    smiles = molecule.to_smiles(isomeric=False, explicit_hydrogens=True)
    # if len(smiles) < 65:
    grouped_molecules[smiles].append((record, molecule))

# Create targets directory and write the input files
Path("./abinitio_targets").mkdir(parents=True, exist_ok=True)
targets_input = open('./abinitio_targets/abinitio_targets.in', 'w')
targets_excluded = open('./abinitio_targets/targets_with_single_conformer_excluded.txt', 'w')

# Iterate over the molecules and the grouped together conformers
i = 0
for key, value in grouped_molecules.items():
    record, molecule = list(value)[0]
    # # Don't need this since we are picking from optimized trajectories
    # Check whether there are at least two conformers to take energy difference
    # if len(value) < 2:
    #     print("Single molecule, need at least two conformers for relative energy, ignoring this record", key)
    #     targets_excluded.write(f"QCA Record ID:{record.id}, SMILES: {molecule.to_smiles(mapped=True)}\n")
    #     continue
    i = i + 1

    target_dir = './abinitio_targets/Abinitio_QCA-' + str(i) + '-' + molecule.hill_formula
    name = os.path.basename(target_dir)
    # Write targets to a file that can be used in optimize_debug.in
    targets_input.write(f"$target \n"
                        f"  name {name}\n"
                        f"  type AbInitio_SMIRNOFF\n"
                        f"  mol2 mol-{i}.sdf\n"
                        f"  pdb mol-{i}.pdb\n"
                        f"  coords mol-{i}.xyz\n"
                        f"  writelevel 2\n"
                        f"  energy 1\n"
                        f"  force 1\n" 
                        f"  w_energy 1.0\n"
                        f"  w_force 0.1\n"
                        f"  fitatoms 0\n" ## all atoms
                        f"  energy_mode qm_minimum\n" ## 'average', 'qm_minimum', or 'absolute'
                        f"  energy_asymmetry 2.0\n" ## Assign a greater weight to MM snapshots that underestimate the QM energy (surfaces referenced to QM absolute minimum)
                        f"  attenuate 1\n"
                        f"  energy_denom 1.0\n"
                        f"  energy_upper 5.0\n"
                        f"  openmm_platform Reference\n"
                        f"$end\n"
                        f"\n")

    # Create target directory for this molecule Abinitio_QCA-ID-hill_formula

    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # Extract energies and gradients from the optimization records
    energies = []
    record_ids = []
    for item in value:
        record_ids.append(item[0].trajectory[-1])
        energies.append(item[0].get_final_energy())
        # Picking some off-equlibrium geometries that will have
        traj_ener = item[0].energies
        indices = [idx for idx, val in enumerate(traj_ener) if val > traj_ener[-1] and val < (traj_ener[-1]+ (2/au2kcal))]
        near_two_kcal_index = min(indices) # Since this is a trajectory barring any bumps in the optimization the energy
                                           # would fall, so picking the index that is low, which means higher in energy
        if len(indices) > 2:
            midway_index = int(np.median(indices))
            energies.append(traj_ener[midway_index])
            record_ids.append(item[0].trajectory[midway_index])
        energies.append(traj_ener[near_two_kcal_index])
        record_ids.append(item[0].trajectory[near_two_kcal_index])

    energies = np.array(energies) # in atomic units
    queried_records = client.query_results(record_ids)
    final_records = []
    for id in record_ids:
        for item in queried_records:
            if item.id == id:
                final_records.append(item)

    record_ids = np.array(record_ids)
    gradients = []
    xyzs = []
    for item in final_records:
        gradients.append(np.array(item.extras["qcvars"]["SCF TOTAL GRADIENT"])) # in atomic units
        xyzs.append(np.array(item.get_molecule().geometry) * unit.bohr.conversion_factor_to(unit.angstrom))
    gradients = np.array(gradients)
    xyzs = np.array(xyzs)


    # Create a FB molecule object from the QCData molecule.
    fb_molecule = FBMolecule()
    fb_molecule.Data = {
        "resname": ["UNK"] * molecule.n_atoms,
        "resid": [0] * molecule.n_atoms,
        "elem": [atom.element.symbol for atom in molecule.atoms],
        "bonds": [
            (bond.atom1_index, bond.atom2_index) for bond in molecule.bonds
        ],
        "name": ', '.join(record_ids[np.argsort(energies)]),
        "xyzs": list(xyzs[np.argsort(energies)]),
        "qm_grads": list(gradients[np.argsort(energies)]),
        "qm_energies": list(np.sort(energies)),
        "comms": [f'QCA ID: {id}, Energy = {ener} hartree'for id, ener in zip(record_ids[np.argsort(energies)], np.sort(energies))]
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

    molecule._conformers = list()
    molecule._properties["SMILES"] = molecule.to_smiles(mapped=True)
    molecule.add_conformer(fb_molecule.Data["xyzs"][0] * unit.angstrom)
    molecule.to_file(sdf_file_output, file_format='sdf')
    fb_molecule.Data["xyzs"] = [fb_molecule.Data["xyzs"][0]]
    del fb_molecule.Data["qm_energies"]
    del fb_molecule.Data["qm_grads"]
    del fb_molecule.Data["comms"]
    fb_molecule.write(pdb_file_output)
    # break

targets_input.close()
