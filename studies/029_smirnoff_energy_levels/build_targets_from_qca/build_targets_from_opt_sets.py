import numpy as np
from openff.qcsubmit.results import OptimizationResultCollection
from collections import defaultdict
import pickle
from openeye import oechem
from pathlib import Path
import os

sage_training_set = OptimizationResultCollection.parse_file(
    '/home/maverick/Desktop/OpenFF/openff-sage/data-set-curation/quantum-chemical/data-sets/1-2-0-opt-set-v3.json')

records_and_molecules = sage_training_set.to_records()

with open('filename.pickle', 'wb') as handle:
    pickle.dump(records_and_molecules, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('filename.pickle', 'rb') as handle:
    records_and_molecules = pickle.load(handle)


grouped_molecules = defaultdict(list)

# Group conformers together by SMILES
for record, molecule in records_and_molecules:
    molecule = molecule.canonical_order_atoms()
    smiles = molecule.to_smiles(isomeric=False, explicit_hydrogens=True)
    grouped_molecules[smiles].append((record, molecule))


Path("./targets").mkdir(parents=True, exist_ok=True)
targets_input = open('./targets/energy_levels.in', 'w')
targets_excluded = open('./targets/targets_with_single_conformer_excluded.txt', 'w')

i = 0
for key, value in grouped_molecules.items():
    # Check whether there are at least two conformers to take energy difference
    if len(value) < 2:
        print("Single molecule, need at least two conformers for energy levels target, ignoring this record", key)
        record, molecule = list(value)[0]
        targets_excluded.write(f"QCA Record ID:{record.id}, SMILES: {molecule.to_smiles(mapped=True)}\n")
        continue
    i = i + 1

    target_dir = './targets/EL_QCA-' + str(i) + '-' + molecule.hill_formula
    name = os.path.basename(target_dir)
    # Add target to optimize.in
    targets_input.write(f"$target \n"
                        f"  name {name}\n"
                        f"  type EnergyLevels_SMIRNOFF\n"
                        f"  mol2 mol-{i}.sdf\n"
                        f"  pdb mol-{i}.pdb\n"
                        f"  coords mol-{i}.xyz\n"
                        f"  writelevel 2\n"
                        f"  attenuate 0\n"
                        f"  energy_denom 1.0\n"
                        f"  energy_upper 5.0\n"
                        f"  openmm_platform Reference\n"
                        f"$end\n"
                        f"\n")

    # Create target directory for this molecule QCA-#_of_mol-hill_formula

    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # Extract energies from the optimization records
    energies = []
    for item in value:
        energies.append(item[0].get_final_energy())
    energies = np.array(energies)

    # Write XYZ file in sorted energies order and sdf, pdb files of a single conformer for FB topologies
    oe_output_file_stream = oechem.oemolostream(target_dir+'/mol-'+str(i)+'.xyz')
    oe_sdf_file_stream = oechem.oemolostream(target_dir + '/mol-' + str(i) + '.sdf')
    oe_pdb_file_stream = oechem.oemolostream(target_dir + '/mol-' + str(i) + '.pdb')

    # Write optgeo_options file within the energy level target directory
    fname = open(target_dir + '/optgeo_options.txt', 'w')

    fname.write("$global \n"
                "  bond_denom 0.01\n"
                "  angle_denom 3\n"
                "  dihedral_denom 10\n"
                "  improper_denom 10\n"
                "$end \n")
    fname.write(f"$target \n"
                f"  name {name}\n"
                f"  geometry mol-{i}.pdb\n "
                f"  topology mol-{i}.xyz\n "
                f"  mol2 mol-{i}.sdf\n"
                f"$end\n")
    qdata = open(target_dir + '/qdata.txt', 'w')

    # Write the conformer files
    for j, index in enumerate(np.argsort(energies)):
        qdata.write(f"ENERGY {energies[index]}\n")
        record, molecule = list(value)[index]
        oemol = molecule.to_openeye()
        oemol.SetTitle(f"QCA Record ID:{record.id}, Energy:{energies[index]}")

        for mol in oemol.GetConfs():
            oechem.OEWriteMolecule(oe_output_file_stream, mol)

        if j == 0:
            for mol in oemol.GetConfs():
                oechem.OESetSDData(mol, "SMILES", molecule.to_smiles(mapped=True))
                oechem.OEWriteMolecule(oe_sdf_file_stream, mol)
                oechem.OEWriteMolecule(oe_pdb_file_stream, mol)
                break

    qdata.close()
    fname.close()

targets_input.close()



