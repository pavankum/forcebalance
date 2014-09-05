""" @package forcebalance.hydration Hydration free energy fitting module

@author Lee-Ping Wang
@date 09/2014
"""

import os
import shutil
import numpy as np
from copy import deepcopy
from forcebalance.target import Target
from forcebalance.molecule import Molecule
from re import match, sub
from forcebalance.finite_difference import fdwrap, f1d2p, f12d3p, in_fd
from collections import defaultdict, OrderedDict
from forcebalance.nifty import getWorkQueue, queue_up, LinkFile, printcool, link_dir_contents, lp_dump, lp_load, _exec, kb, col, flat, uncommadash

from forcebalance.output import getLogger
logger = getLogger(__name__)

class Hydration(Target):

    """ Subclass of Target for fitting force fields to hydration free energies."""
    
    def __init__(self,options,tgt_opts,forcefield):
        """Initialization."""
        
        # Initialize the SuperClass!
        super(Hydration,self).__init__(options,tgt_opts,forcefield)
        
        #======================================#
        # Options that are given by the parser #
        #======================================#
        self.set_option(tgt_opts,'hfedata_txt','datafile')
        self.set_option(tgt_opts,'hfemode')
        # Normalize the weights for molecules in this target
        self.set_option(tgt_opts,'normalize')
        # Energy denominator for evaluating this target
        self.set_option(tgt_opts,'energy_denom','denom')
        # Number of time steps in the liquid "equilibration" run
        self.set_option(tgt_opts,'liquid_eq_steps',forceprint=True)
        # Number of time steps in the liquid "production" run
        self.set_option(tgt_opts,'liquid_md_steps',forceprint=True)
        # Time step length (in fs) for the liquid production run
        self.set_option(tgt_opts,'liquid_timestep',forceprint=True)
        # Time interval (in ps) for writing coordinates
        self.set_option(tgt_opts,'liquid_interval',forceprint=True)
        # Number of time steps in the gas "equilibration" run
        self.set_option(tgt_opts,'gas_eq_steps',forceprint=True)
        # Number of time steps in the gas "production" run
        self.set_option(tgt_opts,'gas_md_steps',forceprint=True)
        # Time step length (in fs) for the gas production run
        self.set_option(tgt_opts,'gas_timestep',forceprint=True)
        # Time interval (in ps) for writing coordinates
        self.set_option(tgt_opts,'gas_interval',forceprint=True)
        # Single temperature for calculating hydration free energies
        self.set_option(tgt_opts,'hfe_temperature',forceprint=True)
        # Single pressure for calculating hydration free energies
        self.set_option(tgt_opts,'hfe_pressure',forceprint=True)
        # Whether to save trajectories (0 = never, 1 = delete after good step, 2 = keep all)
        self.set_option(tgt_opts,'save_traj')
        # Optimize only a subset of the 
        self.set_option(tgt_opts,'subset')
        # List of trajectory files that may be deleted if self.save_traj == 1.
        self.last_traj = []
        # Extra files to be copied back at the end of a run.
        self.extra_output = []
        
        #======================================#
        #     Variables which are set here     #
        #======================================#
        ## The vdata.txt file that contains the hydrations.
        self.datafile = os.path.join(self.tgtdir,self.datafile)
        ## Scripts to be copied from the ForceBalance installation directory.
        self.scripts += ['md_ism_hfe.py']
        ## Read in the reference data
        self.read_reference_data()
        ## Set engname in OptionDict, which gets printed to disk.
        ## This is far from an ideal solution...
        self.OptionDict['engname'] = self.engname
        ## Copy target options into engine options.
        self.engine_opts = OrderedDict(self.OptionDict.items() + options.items())
        del self.engine_opts['name']
        ## Carry out necessary operations for specific modes.
        if self.hfemode.lower() in ['sp', 'single']:
            logger.info("Hydration free energies will be calculated from \n"
                        "geometry optimization and single point energy evaluation\n")
            self.build_engines()
        elif self.hfemode.lower() == 'ti2':
            logger.info("Hydration free energies will be calculated from two-point \n"
                        "thermodynamic integration (linear response approximation)\n")
        else:
            logger.error("Please choose hfemode from single, sp, or ti2\n")
            raise RuntimeError

        if self.FF.rigid_water:
            logger.error('This class cannot be used with rigid water molecules.\n')
            raise RuntimeError

    def read_reference_data(self):
        """ Read the reference hydrational data from a file. """
        self.refdata = OrderedDict([(l.split()[0], float(l.split()[1])) for l in open(self.datafile).readlines()])
        self.molecules = OrderedDict([(i, os.path.abspath(os.path.join(self.root, self.tgtdir, 'molecules', i+self.crdsfx))) for i in self.refdata.keys()])
        for fnm, path in self.molecules.items():
            if not os.path.isfile(path):
                logger.error('Coordinate file %s does not exist!\nMake sure coordinate files are in the right place\n' % path)
                raise RuntimeError
        if self.subset != None:
            subset = uncommadash(self.subset)
            self.whfe = np.array([1 if i in subset else 0 for i in range(len(self.refdata.keys()))])
        else:
            self.whfe = np.ones(len(self.refdata.keys()))

    def run_simulation(self, label, liq, AGrad=True):
        """ 
        Submit a simulation to the Work Queue or run it locally.

        Inputs:
        label = The name of the molecule (and hopefully the folder name that you're running in)
        liq = True/false flag indicating whether to run in liquid or gas phase
        """
        wq = getWorkQueue()

        # Create a dictionary of MD options that the script will read.
        md_opts = OrderedDict()
        md_opts['temperature'] = self.hfe_temperature
        md_opts['pressure'] = self.hfe_pressure
        md_opts['minimize'] = True
        if liq: 
            sdnm = 'liq'
            md_opts['nequil'] = self.liquid_eq_steps
            md_opts['nsteps'] = self.liquid_md_steps
            md_opts['timestep'] = self.liquid_timestep
            md_opts['sample'] = self.liquid_interval
        else: 
            sdnm = 'gas'
            md_opts['nequil'] = self.gas_eq_steps
            md_opts['nsteps'] = self.gas_md_steps
            md_opts['timestep'] = self.gas_timestep
            md_opts['sample'] = self.gas_interval

        eng_opts = deepcopy(self.engine_opts)
        # Enforce implicit solvent in the liquid simulation.
        # We need to be more careful with this when running explicit solvent. 
        eng_opts['implicit_solvent'] = liq
        eng_opts['coords'] = os.path.basename(self.molecules[label])

        os.makedirs(sdnm)
        os.chdir(sdnm)
        if not os.path.exists('md_result.p'):
            # Link in a bunch of files... what were these again?
            link_dir_contents(os.path.join(self.root,self.rundir),os.getcwd())
            # Link in the scripts required to run the simulation
            for f in self.scripts:
                LinkFile(os.path.join(os.path.split(__file__)[0],"data",f),os.path.join(os.getcwd(),f))
            # Link in the coordinate file.
            LinkFile(self.molecules[label], './%s' % os.path.basename(self.molecules[label]))
            # Store names of previous trajectory files.
            self.last_traj += [os.path.join(os.getcwd(), i) for i in self.extra_output]
            # Write target, engine and simulation options to disk.
            lp_dump((self.OptionDict, eng_opts, md_opts), 'simulation.p')
            # Execute the script for running molecular dynamics.
            cmdstr = '%s python md_ism_hfe.py %s' % (self.prefix, "-g" if AGrad else "")
            if wq == None:
                logger.info("Running condensed phase simulation locally.\n")
                logger.info("You may tail -f %s/npt.out in another terminal window\n" % os.getcwd())
                _exec(cmdstr, copy_stderr=True, outfnm='md.out')
            else:
                queue_up(wq, command = cmdstr+' &> md.out',
                         input_files = self.scripts + ['simulation.p', 'forcefield.p', os.path.basename(self.molecules[label])],
                         output_files = ['md_result.p', 'md.out'] + self.extra_output, tgt=self)
        os.chdir('..')

    def submit_liq_gas(self, mvals, AGrad=True):
        """
        Set up and submit/run sampling simulations in the liquid and gas phases.
        """
        # This routine called by Objective.stage() will run before "get".
        # It submits the jobs to the Work Queue and the stage() function will wait for jobs to complete.
        printcool("Target: %s - launching %i MD simulations\nTime steps (liq):" 
                  "%i (eq) + %i (md)\nTime steps (g): %i (eq) + %i (md)" % 
                  (self.name, 2*len(self.refdata.keys()), self.liquid_eq_steps, self.liquid_md_steps,
                   self.gas_eq_steps, self.gas_md_steps), color=0)
        # If self.save_traj == 1, delete the trajectory files from a previous good optimization step.
        if self.evaluated and self.goodstep and self.save_traj < 2:
            for fn in self.last_traj:
                if os.path.exists(fn):
                    os.remove(fn)
        self.last_traj = []
        # Set up and run the NPT simulations.
        # Less fully featured than liquid simulation; NOT INCLUDED are
        # 1) Temperature and pressure
        # 2) Multiple initial conditions
        for label in self.refdata.keys():
            if not os.path.exists(label):
                os.makedirs(label)
            os.chdir(label)
            # Run liquid and gas phase simulations.
            self.run_simulation(label, 0, AGrad)
            self.run_simulation(label, 1, AGrad)
            os.chdir('..')

    def submit_jobs(self, mvals, AGrad=True, AHess=True):
        # If not calculating HFE using simulations, exit this function.
        if self.hfemode.lower() not in ['ti2']:
            return
        else:
            # Prior to running simulations, write the force field pickle
            # file which will be shared by all simulations.
            self.serialize_ff(mvals)
        if self.hfemode.lower() in ['ti2']:
            self.submit_liq_gas(mvals, AGrad)

    def build_engines(self):
        """ Create a list of engines which are used to calculate HFEs using single point evaluation. """
        self.engines = OrderedDict()
        self.liq_engines = OrderedDict()
        self.gas_engines = OrderedDict()
        for mnm in self.refdata.keys():
            pdbfnm = os.path.abspath(os.path.join(self.root,self.tgtdir, 'molecules', mnm+'.pdb'))
            self.liq_engines[mnm] = self.engine_(target=self, coords=pdbfnm, implicit_solvent=True, **self.engine_opts)
            self.gas_engines[mnm] = self.engine_(target=self, coords=pdbfnm, implicit_solvent=False, **self.engine_opts)

    def indicate(self):
        """ Print qualitative indicator. """
        banner = "Hydration free energies (kcal/mol)"
        headings = ["Molecule", "Reference", "Calculated", "Difference", "Weight", "Residual"]
        data = OrderedDict([(i, ["%.4f" % self.refdata[i], "%.4f" % self.calc[i], "%.4f" % (self.calc[i] - self.refdata[i]), 
                                 "%.4f" % self.whfe[ii], "%.4f" % (self.whfe[ii]*(self.calc[i] - self.refdata[i])**2)]) for ii, i in enumerate(self.refdata.keys())])
        self.printcool_table(data, headings, banner)

    def hydration_driver_sp(self):
        """ Calculate HFEs using single point evaluation. """
        hfe = OrderedDict()
        for mnm in self.refdata.keys():
            eliq, rmsdliq = self.liq_engines[mnm].optimize()
            egas, rmsdgas = self.gas_engines[mnm].optimize()
            hfe[mnm] = eliq - egas
        return hfe

    def get_sp(self, mvals, AGrad=False, AHess=False):
        """ Get the hydration free energy and first parameteric derivatives using single point energy evaluations. """
        def get_hfe(mvals_):
            self.FF.make(mvals_)
            self.hfe_dict = self.hydration_driver_sp()
            return np.array(self.hfe_dict.values())
        calc_hfe = get_hfe(mvals)
        D = calc_hfe - np.array(self.refdata.values())
        dD = np.zeros((self.FF.np,len(self.refdata.keys())))
        if AGrad or AHess:
            for p in self.pgrad:
                dD[p,:], _ = f12d3p(fdwrap(get_hfe, mvals, p), h = self.h, f0 = calc_hfe)
        return D, dD

    def get_ti2(self, mvals, AGrad=False, AHess=False):
        """ Get the hydration free energy using two-point thermodynamic integration. """
        self.hfe_dict = OrderedDict()
        dD = np.zeros((self.FF.np,len(self.refdata.keys())))
        beta = 1. / (kb * self.hfe_temperature)
        for ilabel, label in enumerate(self.refdata.keys()):
            os.chdir(label)
            # This dictionary contains observables keyed by each phase.
            data = defaultdict(dict)
            for p in ['gas', 'liq']:
                os.chdir(p)
                # Load the results from molecular dynamics.
                results = lp_load('md_result.p')
                # Time series of hydration energies.
                H = results['Hydration']
                # Store the average hydration energy.
                data[p]['Hyd'] = np.mean(H)
                if AGrad:
                    dE = results['Potential_Derivatives']
                    dH = results['Hydration_Derivatives']
                    # Calculate the parametric derivative of the average hydration energy.
                    data[p]['dHyd'] = np.mean(dH,axis=1)-beta*(flat(np.matrix(dE)*col(H)/len(H))-np.mean(dE,axis=1)*np.mean(H))
                os.chdir('..')
            # Calculate the hydration free energy as the average of liquid and gas hydration energies.
            # Note that the molecular dynamics methods return energies in kJ/mol.
            self.hfe_dict[label] = 0.5*(data['liq']['Hyd']+data['gas']['Hyd']) / 4.184
            if AGrad:
                # Calculate the derivative of the hydration free energy.
                dD[:, ilabel] = 0.5*self.whfe[ilabel]*(data['liq']['dHyd']+data['gas']['dHyd']) / 4.184
            os.chdir('..')
        calc_hfe = np.array(self.hfe_dict.values())
        D = self.whfe*(calc_hfe - np.array(self.refdata.values()))
        return D, dD

    def get(self, mvals, AGrad=False, AHess=False):
        """ Evaluate objective function. """
        Answer = {'X':0.0, 'G':np.zeros(self.FF.np), 'H':np.zeros((self.FF.np, self.FF.np))}
        if self.hfemode.lower() == 'single' or self.hfemode.lower() == 'sp':
            D, dD = self.get_sp(mvals, AGrad, AHess)
        elif self.hfemode.lower() == 'ti2':
            D, dD = self.get_ti2(mvals, AGrad, AHess)
        Answer['X'] = np.dot(D,D) / self.denom**2 / (np.sum(self.whfe) if self.normalize else 1)
        for p in self.pgrad:
            Answer['G'][p] = 2*np.dot(D, dD[p,:]) / self.denom**2 / (np.sum(self.whfe) if self.normalize else 1)
            for q in self.pgrad:
                Answer['H'][p,q] = 2*np.dot(dD[p,:], dD[q,:]) / self.denom**2 / (np.sum(self.whfe) if self.normalize else 1)
        if not in_fd():
            self.calc = self.hfe_dict
            self.objective = Answer['X']
        return Answer
