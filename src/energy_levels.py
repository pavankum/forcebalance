""" @package forcebalance.energy_levels Energy levels fitting module.

@author Pavan Behara
@date 10/2022

Modification of TorsionProfile target to use for fitting ranked energy levels from optimized conformers.
"""
from __future__ import division

import os
from collections import OrderedDict
from copy import deepcopy
from itertools import combinations
import numpy as np
from forcebalance.finite_difference import f12d3p, fdwrap, in_fd
from forcebalance.molecule import Molecule
from forcebalance.nifty import eqcgmx, printcool_dictionary, warn_press_key
from forcebalance.optimizer import Counter
from forcebalance.output import getLogger
from forcebalance.opt_geo_target import compute_rmsd

logger = getLogger(__name__)
from forcebalance.target import Target

RADIAN_2_DEGREE = 180 / np.pi

class EnergyLevelsTarget(Target):
    """ Subclass of Target for fitting MM optimized geometries to QM optimized geometries. """

    def __init__(self, options, tgt_opts, forcefield):
        super(EnergyLevelsTarget, self).__init__(options, tgt_opts, forcefield)

        ## Read in the coordinate files and get topology information from PDB
        if hasattr(self, 'pdb') and self.pdb is not None:
            self.mol = Molecule(os.path.join(self.root, self.tgtdir, self.coords),
                                top=(os.path.join(self.root, self.tgtdir, self.pdb)))
        else:
            self.mol = Molecule(os.path.join(self.root, self.tgtdir, self.coords))
        ## Number of snapshots.
        self.ns = len(self.mol)
        ## Option for how much data to write to disk.
        self.set_option(tgt_opts, 'writelevel', 'writelevel')
        self.set_option(None, None, 'optgeo_options', os.path.join(self.tgtdir, tgt_opts['optgeo_options_txt']))
        self.sys_opts = self.parse_optgeo_options(self.optgeo_options)
        ## Attenuate the weights as a function of energy
        self.set_option(tgt_opts, 'attenuate', 'attenuate')
        ## Calculate internal coordinates or not
        self.set_option(tgt_opts, 'calc_ic', 'calc_ic')
        ## Harmonic restraint for non-torsion atoms in kcal/mol.
        self.set_option(tgt_opts, 'restrain_k', 'restrain_k')
        ## Energy denominator for objective function
        self.set_option(tgt_opts, 'energy_denom', 'energy_denom')
        ## Set upper cutoff energy
        self.set_option(tgt_opts, 'energy_upper', 'energy_upper')
        ## Set e_width in the switching function
        self.set_option(tgt_opts, 'e_width', 'e_width')
        ## Read in the reference data.
        self.read_reference_data()
        ## Create internal coordinates
        if self.calc_ic:
            self._build_internal_coordinates()
        self._setup_scale_factors()
        ## Build keyword dictionaries to pass to engine.
        engine_args = OrderedDict(list(self.OptionDict.items()) + list(options.items()))
        engine_args.pop('name', None)
        ## Create engine object.
        self.engine = self.engine_(target=self, mol=self.mol, **engine_args)

    def _build_internal_coordinates(self):
        "Build internal coordinates system with geometric.internal.PrimitiveInternalCoordinates"
        # geometric module is imported to build internal coordinates
        # importing here will avoid import error for calculations not using this target
        from geometric.internal import PrimitiveInternalCoordinates, Distance, Angle, Dihedral, OutOfPlane
        self.internal_coordinates = OrderedDict()
        print("Building internal coordinates using topologys")
        p_IC = PrimitiveInternalCoordinates(self.mol)
        for i in range(self.ns):
            # logger.info("Building internal coordinates from file: %s\n" % topfile)
            # here we explicitly pick the bonds, angles and dihedrals to evaluate
            ic_bonds, ic_angles, ic_dihedrals, ic_impropers = [], [], [], []
            for ic in p_IC.Internals:
                if isinstance(ic, Distance):
                    ic_bonds.append(ic)
                elif isinstance(ic, Angle):
                    ic_angles.append(ic)
                elif isinstance(ic, Dihedral):
                    ic_dihedrals.append(ic)
                elif isinstance(ic, OutOfPlane):
                    ic_impropers.append(ic)
            # compute and store reference values
            pos_ref = self.mol.xyzs[i]
            # keep track of the total number of internal coords
            vref_bonds = np.array([ic.value(pos_ref) for ic in ic_bonds])
            self.n_bonds = len(vref_bonds)
            vref_angles = np.array([ic.value(pos_ref)*RADIAN_2_DEGREE for ic in ic_angles])
            self.n_angles = len(vref_angles)
            vref_dihedrals = np.array([ic.value(pos_ref)*RADIAN_2_DEGREE for ic in ic_dihedrals])
            self.n_dihedrals = len(vref_dihedrals)
            vref_impropers = np.array([ic.value(pos_ref)*RADIAN_2_DEGREE for ic in ic_impropers])
            self.n_impropers = len(vref_impropers)
            self.internal_coordinates[i] = {
                'ic_bonds': ic_bonds,
                'ic_angles': ic_angles,
                'ic_dihedrals': ic_dihedrals,
                'ic_impropers': ic_impropers,
                'vref_bonds': vref_bonds,
                'vref_angles': vref_angles,
                'vref_dihedrals': vref_dihedrals,
                'vref_impropers': vref_impropers,
            }

    def get_internal_coords(self, shot, positions):
        """
        calculate the internal coord values for the current positions.
        """
        ic_dict = self.internal_coordinates[shot]
        v_ic = {
        'bonds': np.array([ic.value(positions) for ic in ic_dict['ic_bonds']]),
        'angles': np.array([ic.value(positions)*RADIAN_2_DEGREE for ic in ic_dict['ic_angles']]),
        'dihedrals': np.array([ic.value(positions)*RADIAN_2_DEGREE for ic in ic_dict['ic_dihedrals']]),
        'impropers': np.array([ic.value(positions)*RADIAN_2_DEGREE for ic in ic_dict['ic_impropers']]),
        }
        return v_ic

    @staticmethod
    def parse_optgeo_options(filename):
        """ Parse an optgeo_options.txt file into specific OptGeoTarget Target Options"""
        logger.info("Reading optgeo options from file: %s\n" % filename)
        global_opts = OrderedDict()
        sys_opts = OrderedDict()
        section = None
        section_opts = OrderedDict()
        with open(filename) as f:
            for ln, line in enumerate(f, 1):
                # Anything after "#" is a comment
                line = line.split("#", maxsplit=1)[0].strip()
                if not line: continue
                ls = line.split()
                key = ls[0].lower()
                if key[0] == "$":
                    # section sign $
                    if key == '$end':
                        if section is None:
                            warn_press_key("Line %i: Encountered $end before any section." % ln)
                        elif section == 'global':
                            # global options read finish
                            global_opts = section_opts
                        elif section == 'system':
                            # check if system section contains name
                            if 'name' not in section_opts:
                                warn_press_key("Line %i: You need to specify a name for the system section ending." % ln)
                            elif section_opts['name'] in sys_opts:
                                warn_press_key("Line %i: A system named %s already exists in Systems" % (ln, section_opts['name']))
                            else:
                                sys_opts[section_opts['name']] = section_opts
                        section = None
                        section_opts = OrderedDict()
                    else:
                        if section is not None:
                            warn_press_key("Line %i: Encountered section start %s before previous section $end." % (ln, key))
                        if key == '$global':
                            section = 'global'
                        elif key == '$system':
                            section = 'system'
                        else:
                            warn_press_key("Line %i: Encountered unsupported section name %s " % (ln, key))
                else:
                    # put normal key-value options into section_opts
                    if key in ['name', 'geometry', 'topology']:
                        if len(ls) != 2:
                            warn_press_key("Line %i: one value expected for key %s" % (ln, key))
                        if section == 'global':
                            warn_press_key("Line %i: key %s should not appear in $global section" % (ln, key))
                        section_opts[key] = ls[1]
                    elif key in ['bond_denom', 'angle_denom', 'dihedral_denom', 'improper_denom']:
                        if len(ls) != 2:
                            warn_press_key("Line %i: one value expected for key %s" % (ln, key))
                        section_opts[key] = float(ls[1])
                    elif key == 'mol2':
                        # special parsing for mol2 option for SMIRNOFF engine
                        # the value is a list of filenames
                        section_opts[key] = ls[1:]
        # apply a few default global options
        global_opts.setdefault('bond_denom', 0.02)
        global_opts.setdefault('angle_denom', 3)
        global_opts.setdefault('dihedral_denom', 10.0)
        global_opts.setdefault('improper_denom', 10.0)
        # copy global options into each system
        for sys_name, sys_opt_dict in sys_opts.items():
            for k,v in global_opts.items():
                # do not overwrite system options
                sys_opt_dict.setdefault(k, v)
            for k in ['name', 'geometry', 'topology']:
                if k not in sys_opt_dict:
                    warn_press_key("key %s missing in system section named %s" %(k, sys_name))
        return sys_opts

    def _setup_scale_factors(self, bond_denom=0.0, angle_denom=0, dihedral_denom=100, improper_denom=0):
        self.scale_bond = 1.0 / bond_denom if bond_denom != 0 else 0.0
        self.scale_angle = 1.0 / angle_denom if angle_denom != 0 else 0.0
        self.scale_dihedral = 1.0 / dihedral_denom if dihedral_denom != 0 else 0.0
        self.scale_improper = 1.0 / improper_denom if improper_denom != 0 else 0.0

    def read_reference_data(self):

        """ Read the reference ab initio data from a file such as qdata.txt.

        After reading in the information from qdata.txt, it is converted
        into the GROMACS energy units (kind of an arbitrary choice).
        """
        ## Reference (QM) energies
        self.eqm = []
        ## The qdata.txt file that contains the QM energies and forces
        self.qfnm = os.path.join(self.tgtdir, "qdata.txt")
        # Parse the qdata.txt file
        for line in open(os.path.join(self.root, self.qfnm)):
            sline = line.split()
            if len(sline) == 0:
                continue
            elif sline[0] == 'ENERGY':
                self.eqm.append(float(sline[1]))

        if len(self.eqm) != self.ns:
            raise RuntimeError("Length of qdata.txt should match number of structures")

        # Turn everything into arrays, convert to kcal/mol
        self.eqm = np.array(self.eqm)
        self.eqm *= eqcgmx / 4.184
        # Use the minimum energy structure of the QM as reference
        self.eqm -= np.min(self.eqm)
        self.smin = np.argmin(self.eqm)
        logger.info("Referencing all energies to the snapshot %i (minimum energy structure in QM)\n" % self.smin)



    def indicate(self):
        title_str = "Energy Levels: %s, Objective = % .5e, Units = kcal/mol, Angstrom" % (self.name, self.objective)
        # LPW: This title is carefully placed to align correctly
        column_head_str1 = "%-50s %-10s %-12s %-18s %-12s %-10s %-11s %-10s" % (
        "System", "Min(QM) - ", "Max(QM)", "Min(MM) - ", "Max(MM)", "Max-RMSD", "ddE", "Obj-Fn")
        printcool_dictionary(self.PrintDict, title=title_str + '\n' + column_head_str1, keywidth=50,
                             center=[True, False])

    def get(self, mvals, AGrad=False, AHess=False):
        from forcebalance.opt_geo_target import periodic_diff

        Answer = {'X': 0.0, 'G': np.zeros(self.FF.np), 'H': np.zeros((self.FF.np, self.FF.np))}
        self.PrintDict = OrderedDict()

        def switching_function(x, w):
            return 0.5 + 0.5 * np.tanh(x/w)

        def compute(mvals_, indicate=False):
            self.FF.make(mvals_)
            M_opts = None
            compute.emm = []
            compute.rmsd = []
            compute.total_ic_diff = []
            all_diff = {}
            all_rmsd = {}
            # Adding RMSD + ddE
            for i in range(self.ns):
                energy, rmsd, M_opt = self.engine.optimize(shot=i, align=False)
                # Create a molecule object to hold the MM-optimized structures
                compute.emm.append(energy)
                compute.rmsd.append(rmsd)
                # extract the final geometry and calculate the internal coords after optimization
                opt_pos = self.engine.getContextPosition()
                if self.calc_ic:
                    v_ic = self.get_internal_coords(shot=i, positions=opt_pos)
                    # get the reference values in internal coords
                    vref_bonds = self.internal_coordinates[i]['vref_bonds']
                    vref_angles = self.internal_coordinates[i]['vref_angles']
                    vref_dihedrals = self.internal_coordinates[i]['vref_dihedrals']
                    vref_impropers = self.internal_coordinates[i]['vref_impropers']
                    vtar_bonds = v_ic['bonds']
                    diff_bond = (abs(vref_bonds - vtar_bonds) * self.scale_bond).tolist() if self.n_bonds > 0 else []
                    # print("bonds", diff_bond)
                    # objective contribution from angles
                    vtar_angles = v_ic['angles']
                    diff_angle = (
                                abs(periodic_diff(vref_angles, vtar_angles, 360)) * self.scale_angle).tolist() if self.n_angles > 0 else []
                    # print("angles", diff_angle)
                    # objective contribution from dihedrals
                    vtar_dihedrals = v_ic['dihedrals']
                    diff_dihedral = (abs(periodic_diff(vref_dihedrals, vtar_dihedrals,
                                                   360)) * self.scale_dihedral).tolist() if self.n_dihedrals > 0 else []
                    # print("dihedrals", diff_dihedral)
                    # objective contribution from improper dihedrals
                    vtar_impropers = v_ic['impropers']
                    diff_improper = (abs(periodic_diff(vref_impropers, vtar_impropers,
                                                   360)) * self.scale_improper).tolist() if self.n_impropers > 0 else []
                    # print("impropers", diff_improper)
                    # combine objective values into a big result list
                    sys_obj_list = diff_bond + diff_angle + diff_dihedral + diff_improper
                    # store
                    all_diff[i] = dict(bonds=diff_bond, angle=diff_angle, dihedral=diff_dihedral, improper=diff_improper)
                    # compute the objective for just this conformer and add it to a list
                    compute.total_ic_diff.append(np.dot(sys_obj_list, sys_obj_list))
                    # make a list of rmsd values
                    current_rmsd = dict(bonds=compute_rmsd(vref_bonds, vtar_bonds), angle=compute_rmsd(vref_angles, vtar_angles, v_periodic=360),
                                        dihedral=compute_rmsd(vref_dihedrals, vtar_dihedrals, v_periodic=360), improper=compute_rmsd(vref_impropers, vtar_impropers, v_periodic=360))
                    all_rmsd[i] = current_rmsd

                if M_opts is None:
                    M_opts = deepcopy(M_opt)
                else:
                    M_opts += M_opt
            compute.emm = np.array(compute.emm)
            compute.emm -= compute.emm[self.smin]
            compute.rmsd = np.array(compute.rmsd)
            compute.total_ic_diff = np.array(compute.total_ic_diff)

            if self.attenuate:
                # Attenuate energies by an amount proportional to their
                # value above the minimum.
                eqm1 = self.eqm - np.min(self.eqm)
                denom = self.energy_denom
                upper = self.energy_upper
                self.wts = np.ones(self.ns)
                for i in range(self.ns):
                    if eqm1[i] > upper:
                        self.wts[i] = 0.0
                    elif eqm1[i] < denom:
                        self.wts[i] = 1.0 / denom
                    else:
                        self.wts[i] = 1.0 / np.sqrt(
                            denom ** 2 + (eqm1[i] - denom) ** 2)
            else:
                self.wts = np.ones(self.ns)

            # Normalize weights.
            self.wts /= sum(self.wts)

            if indicate:
                if self.writelevel > 0:
                    energy_comparison = np.array([
                        self.eqm,
                        compute.emm,
                        compute.emm - self.eqm,
                        np.sqrt(self.wts)
                    ]).T
                    np.savetxt("EnergyCompare.txt", energy_comparison,
                               header="%11s  %12s  %12s  %12s" % ("QMEnergy", "MMEnergy", "Delta(MM-QM)", "Weight"),
                               fmt="% 12.6e")
                    M_opts.write('mm_minimized.xyz')
                    try:
                        import matplotlib.pyplot as plt
                        plt.switch_backend('agg')
                        fig, ax = plt.subplots()
                        plt.rcParams['font.size'] = 10
                        dsort = np.argsort(self.eqm)
                        plt.xlim(0, 14)
                        plt.xticks(np.arange(0, 15, 1.0))
                        my_xticks = ['' for i in range(len(ax.get_xticks().tolist()))]
                        my_xticks[1] = 'QM'
                        my_xticks[3] = 'MM Curr.'
                        my_xticks[5] = 'MM Init.'
                        my_xticks[8] = 'RMSD Curr.'
                        my_xticks[11] = 'RMSD Init.'

                        ticks_loc = ax.get_xticks().tolist()
                        ax.xaxis.set_ticks(ticks_loc)
                        ax.set_xticklabels(my_xticks)
                        ax.plot([1] * len(self.eqm), self.eqm[dsort], '_', label='QM', markersize=20, linewidth=2)

                        ax2 = ax.twinx()
                        ax2.set_ylim([0, 1.5])

                        y_seen = []
                        flag = False
                        for xi, yi, tx in zip([1] * len(self.eqm), self.eqm[dsort], np.arange(len(self.eqm))):
                            if y_seen:
                                flag = np.any(np.full(len(y_seen), yi) - y_seen < 0.2)
                            if flag:
                                ax.annotate(tx, xy=(.1 * xi + 0.2, yi), xytext=(7, 3), size=8,
                                            ha="center", va='top', textcoords="offset points")
                            else:
                                ax.annotate(tx, xy=(.1 * xi, yi), xytext=(7, 3), size=8,
                                            ha="center", va='top', textcoords="offset points")
                            y_seen.append(yi)

                        if hasattr(self, 'emm_orig'):
                            ax.plot([3] * len(compute.emm[dsort]), compute.emm[dsort], '_', label='MM Curr.',
                                    markersize=20, linewidth=2)
                            y_seen = []
                            flag = False
                            for xi, yi, tx in zip([3] * len(compute.emm[dsort]), compute.emm[dsort], np.arange(len(self.eqm))):
                                if y_seen:
                                    flag = np.any(np.full(len(y_seen), yi) - y_seen < 0.2)
                                if flag:
                                    ax.annotate(tx, xy=(.6 * xi + 0.2, yi), xytext=(7, 3), size=8,
                                                ha="center", va='top', textcoords="offset points")
                                else:
                                    ax.annotate(tx, xy=(.6 * xi, yi), xytext=(7, 3), size=8,
                                                ha="center", va='top', textcoords="offset points")
                                y_seen.append(yi)
                            ax.plot([5] * len(self.emm_orig[dsort]), self.emm_orig[dsort], '_', label='MM Init.',
                                    markersize=20, linewidth=2)  #
                            y_seen = []
                            flag = False
                            for xi, yi, tx in zip([5] * len(self.emm_orig[dsort]), self.emm_orig[dsort], np.arange(len(self.eqm))):
                                if y_seen:
                                    flag = np.any(np.full(len(y_seen), yi) - y_seen < 0.2)
                                if flag:
                                    ax.annotate(tx, xy=(.8 * xi + 0.2, yi), xytext=(7, 3), size=8,
                                                ha="center", va='top', textcoords="offset points")
                                else:
                                    ax.annotate(tx, xy=(.8 * xi, yi), xytext=(7, 3), size=8,
                                                ha="center", va='top', textcoords="offset points")
                                y_seen.append(yi)

                            #RMSD plots
                            ax2.plot([8] * len(compute.rmsd[dsort]), compute.rmsd[dsort], '_', label='RMSD Curr.',
                                    markersize=20, linewidth=2, color='cyan')
                            y_seen = []
                            flag = False
                            for xi, yi, tx in zip([8] * len(compute.rmsd[dsort]), compute.rmsd[dsort],
                                                  np.arange(len(self.eqm))):
                                if y_seen:
                                    flag = np.any(np.full(len(y_seen), yi) - y_seen < 0.05)
                                if flag:
                                    ax2.annotate(tx, xy=(.8 * xi + 0.2, yi), xytext=(7, 3), size=8,
                                                ha="center", va='top', textcoords="offset points")
                                else:
                                    ax2.annotate(tx, xy=(.8 * xi, yi), xytext=(7, 3), size=8,
                                                ha="center", va='top', textcoords="offset points")
                                y_seen.append(yi)
                            ax2.plot([11] * len(self.rmsd_orig[dsort]), self.rmsd_orig[dsort], '_',
                                    label='RMSD Init.',
                                    markersize=20, linewidth=2, color='magenta')  #
                            y_seen = []
                            flag = False
                            for xi, yi, tx in zip([11] * len(self.rmsd_orig[dsort]), self.rmsd_orig[dsort],
                                                  np.arange(len(self.eqm))):
                                if y_seen:
                                    flag = np.any(np.full(len(y_seen), yi) - y_seen < 0.05)
                                if flag:
                                    ax2.annotate(tx, xy=(.9 * xi + 0.2, yi), xytext=(7, 3), size=8,
                                                ha="center", va='top', textcoords="offset points")
                                else:
                                    ax2.annotate(tx, xy=(.9 * xi, yi), xytext=(7, 3), size=8,
                                                ha="center", va='top', textcoords="offset points")
                                y_seen.append(yi)


                        else:
                            ax.plot([3] * len(compute.emm[dsort]), compute.emm[dsort], '_', label='MM Curr.',
                                    markersize=20, linewidth=2)  #
                            y_seen = []
                            flag = False
                            for xi, yi, tx in zip([3] * len(compute.emm[dsort]), compute.emm[dsort], np.arange(len(self.eqm))):
                                if y_seen:
                                    flag = np.any(np.full(len(y_seen), yi) - y_seen < 0.2)
                                if flag:
                                    ax.annotate(tx, xy=(.6 * xi + 0.2, yi), xytext=(7, 3), size=8,
                                                ha="center", va='top', textcoords="offset points")
                                else:
                                    ax.annotate(tx, xy=(.6 * xi, yi), xytext=(7, 3), size=8,
                                                ha="center", va='top', textcoords="offset points")
                                y_seen.append(yi)
                            self.emm_orig = compute.emm.copy()

                            # RMSD plots
                            ax2.plot([8] * len(compute.rmsd), compute.rmsd[dsort], '_', label='MM RMSD', markersize=20,
                                     color='cyan')
                            y_seen = []
                            flag = False
                            for xi, yi, tx in zip([8] * len(compute.rmsd), compute.rmsd[dsort], np.arange(len(self.eqm))):
                                if y_seen:
                                    flag = np.any(np.full(len(y_seen), yi) - y_seen < 0.05)
                                if flag:
                                    ax2.annotate(tx, xy=(.8 * xi + 0.2, yi), xytext=(7, 3), size=8,
                                                ha="center", va='top', textcoords="offset points")
                                else:
                                    ax2.annotate(tx, xy=(.8 * xi, yi), xytext=(7, 3), size=8,
                                                ha="center", va='top', textcoords="offset points")
                                y_seen.append(yi)

                            self.rmsd_orig = compute.rmsd.copy()

                        handles_, labels_ = ax.get_legend_handles_labels()
                        lgnd = ax.legend(handles=handles_, loc='upper right', prop={'size': 8})
                        # change the marker size manually for both lines
                        for hndle in lgnd.legendHandles:
                            hndle.set_markersize(10)
                            hndle.set_alpha(1)
                        ax.set_xlabel('Energy Levels (ranked by QM energies)')
                        ax.set_ylabel('Energy (kcal/mol)')

                        ax2.set_ylabel('RMSD wrt QM in $\AA$')

                        
                        fig.suptitle('Energy levels: iteration %i\nSystem: %s' % (Counter(), self.name))
                        fig.savefig('plot_energy_levels.pdf')

                    except ImportError:
                        logger.warning("matplotlib package is needed to make torsion profile plots\n")


            # print("ddE contrib.:", (np.sqrt(self.wts)) * (compute.emm - self.eqm), "IC_rmsd:", compute.total_ic_diff)
            return  (np.sqrt(self.wts)) * (compute.emm - self.eqm) #+ 0.1 * (np.sqrt(self.wts)/2) * compute.total_ic_diff

        compute.emm = None
        compute.rmsd = None
        compute.total_ic_diff = None

        V = compute(mvals, indicate=True)

        Answer['X'] = np.dot(V, V)

        # Energy RMSE
        # e_rmse = 0 #np.sqrt(np.dot(self.wts, (compute.emm - self.eqm) ** 2))
        indcs = combinations(range(len(compute.emm)), 2)
        relative_e_err = 0
        for a, b in indcs:
            relative_e_err += (self.eqm[a] - self.eqm[b]) - (compute.emm[a] -
                                                   compute.emm[b])
        uniq_combs = len(compute.emm) * (len(compute.emm) - 1)/2
        e_rmse = (1/uniq_combs) * np.sqrt(np.square(relative_e_err))
        # IC RMSE
        if self.calc_ic:
            r = (np.sqrt(self.wts) / 1 * compute.total_ic_diff)

        self.PrintDict[self.name] = \
            '%6.3f - %-6.3f   % 6.3f - %-6.3f    %6.3f    %7.4f   % 7.4f' % (
        min(self.eqm), max(self.eqm), min(compute.emm), max(compute.emm), max(compute.rmsd), e_rmse, Answer['X'])

        # compute gradients and hessian
        dV = np.zeros((self.FF.np, len(V)))
        if AGrad or AHess:
            for p in self.pgrad:
                dV[p, :], _ = f12d3p(fdwrap(compute, mvals, p), h=self.h, f0=V)

        for p in self.pgrad:
            Answer['G'][p] = 2 * np.dot(V, dV[p, :])
            for q in self.pgrad:
                Answer['H'][p, q] = 2 * np.dot(dV[p, :], dV[q, :])
        if not in_fd():
            self.objective = Answer['X']
            self.FF.make(mvals)
        return Answer

