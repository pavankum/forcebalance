"""This file contains a documentation generating script. Doxygen
is used to do the actual generation, so these functions act primarily to
streamline the process and provide some customizations to the automatically
generated documents.

The motivation is:
- Want to have automatic online docs
- Want to version-control PDF manuals
- Don't want to version-control HTML / latex files (it generates too many changes)

The solution is:
- Have a separate gh-pages branch that only keeps track of latex and html folders, and PDF files
- Pushing to gh-pages branch will update documentation on
  http://leeping.github.io/forcebalance/doc/html/index.html and
  http://leeping.github.io/forcebalance/doc/ForceBalance-Manual.pdf

The workflow for generating documentation:
- Generate doxygen config files from source controlled templates
- Generate option index using make-option-index.py
- Generate doxygen source files from source controlled text files
- Delete existing HTML and latex files
- Run doxygen to generate HTML files and latex files
- Hack the HTML files to add extra tabs
- Run latex to generate PDF documents

If upstream update is requested:
- Commit master branch (because manual is part of master branch)
- Move html and latex folders out of the way, check out gh-pages branch, update html and latex folders
  This is because html and latex folders are not tracked by the master branch,
  and if we check out the gh-pages branch we will get an error.
- Check out updated manuals from master branch
- Commit gh-pages branch and push upstream
- Check out master branch and restore folder locations

How to do this effectively:

- Make sure doxypy executable is in the PATH
- Make sure dot (from graphviz) is in the PATH
- Make sure version numbers are correct in four places: .api.cfg, .doxygen.cfg, header.tex, api_header.tex

The comment syntax below in a docstring will break it:
    Quantity
    ========

    Base class for thermodynamical quantity used for fitting. This can
    be any experimental data that can be calculated as an ensemble
    average from a simulation.

"""
from __future__ import print_function

from builtins import input
import os, sys
import re
import shutil
import subprocess
import argparse
from traceback import print_exc
from socket import gethostname
from datetime import datetime

def build(interactive=False, upstream=False):

    if interactive:
        display = lambda txt : input("$ %s " % txt)
    else:
        display = lambda txt : sys.stdout.write("$ %s\n" % txt)

    print("\n# Build list of documented options")
    display("python make-option-index.py > option_index.txt")
    os.system("python make-option-index.py > option_index.txt")

    # generate pages to be included in general documentation
    mainpage=""
    mainpage+="/**\n\n\\mainpage\n\n"
    for fnm in ["introduction.txt", "installation.txt", "usage.txt", "tutorial.txt", "glossary.txt", "option_index.txt"]:
        page=open(fnm,'r')
        mainpage+=page.read()
    mainpage+="\n\\image latex ForceBalance.pdf \"Logo.\" height=10cm\n\n*/"

    # generate pages to be included in API documentation
    api=""
    api+="/**\n\n"
    for fnm in ["roadmap.txt"]:
        page=open(fnm,'r')
        api+=page.read()
    api+="\n\n*/"

    # First attempt to generate documentation.
    try:
        with open('mainpage.dox','w') as f:
            f.write(mainpage)
        with open('api.dox','w') as f:
            f.write(api)

        # Delete HTML and API documentation folders
        display("rm -rf html latex html_ latex_")
        os.system("rm -rf html latex html_ latex_")

        # Run doxygen to generate general documentation
        print("\n# run doxygen with doxygen.cfg as input to generate general documentation")
        display("doxygen doxygen.cfg")
        if subprocess.call(['doxygen', 'doxygen.cfg']): raise OSError("Doxygen returned nonzero value while working on doxygen.cfg")

        # Run doxygen to generate technical (API) documentation
        print("\n# run doxygen with api.cfg as input to generate API documentation")
        display("doxygen api.cfg")
        if subprocess.call(['doxygen', 'api.cfg']): raise OSError("Doxygen returned nonzero value while working on api.cfg")

        # add_tabs script adjusts html
        print("\n# run add_tabs function to adjust tabs on html generated by doxygen")
        display("python -c 'from makedocumentation import add_tabs; add_tabs()'")
        add_tabs()

        # Compile pdf formats
        print("\n# Copy images referenced in latex files to proper folders")
        display("cp Images/ForceBalance.pdf latex/ && cp Images/ForceBalance.pdf latex/api/")
        if not os.path.exists('latex/api'):
            os.makedirs('latex/api')
        shutil.copy('Images/ForceBalance.pdf','latex/')
        shutil.copy('Images/ForceBalance.pdf','latex/api/')

        print("\n# Compile generated latex documentation into pdf")
        display("cd latex && make")
        os.chdir('latex')
        if subprocess.call(['make']): raise OSError("make returned nonzero value while compiling latex/")
        print("\n# Copy generated pdf up to /doc directory")
        display("cd .. && cp latex/refman.pdf ForceBalance-Manual.pdf")
        os.chdir('..')
        shutil.copy('latex/refman.pdf', 'ForceBalance-Manual.pdf')

        #print "\n#Compile generated latex API documentation into pdf"
        #display("cd latex/api/ && make")
        #os.chdir('latex/api/')
        #if subprocess.call(['make']): raise OSError("make returned nonzero value while compiling latex/api/")
        #print "\n# Copy generated API pdf up to /doc directory"
        #display("cd ../.. && cp latex/api/refman.pdf ForceBalance-API.pdf")
        #os.chdir('../..')
        #shutil.copy('latex/api/refman.pdf', 'ForceBalance-API.pdf')
    except:
        print_exc()
        upstream = False  # since documentation generation failed,
        input("\n# encountered ERROR (above). Documentation could not be generated.")
        sys.exit(1)

    if upstream:
        print("\n# Switch to documentation branch before writing files")

        # Move folders to temporary location prior to branch switch
        for fnm in ["latex", "html"]:
            display("mv %s %s_" % (fnm, fnm))
            os.system("mv %s %s_" % (fnm, fnm))

        # Make sure we only push the current branch
        display("git config --global push.default current")
        os.system("git config --global push.default current")

        input("\n Press a key to COMMIT the master branch (will update manuals).")
        display('git commit -a -m "Automatic documentation generation at %s on %s"' % (gethostname(), datetime.now().strftime("%m-%d-%Y %H:%M")))
        if os.system('git commit -a -m "Automatic documentation generation at %s on %s"' % (gethostname(), datetime.now().strftime("%m-%d-%Y %H:%M"))):
            raise OSError("Error trying to commit files to local master branch")

        # Check out the gh-pages branch
        display("git checkout gh-pages")
        if os.system("git checkout gh-pages"):
            print("\n# encountered ERROR in checking out branch (above).  Please commit files and try again.")
            for fnm in ["latex", "html"]:
                os.system("mv %s_ %s" % (fnm, fnm))
            sys.exit(1)

        # Rsync the newly generated html and latex folders
        display("rsync -a --delete html_/ html")
        os.system("rsync -a --delete html_/ html")
        display("rsync -a --delete latex_/ latex")
        os.system("rsync -a --delete latex_/ latex")
        display("git checkout master ForceBalance-API.pdf")
        os.system("git checkout master ForceBalance-API.pdf")
        display("git checkout master ForceBalance-Manual.pdf")
        os.system("git checkout master ForceBalance-Manual.pdf")
        try:
            # Commit the new html and latex files
            print("\n# Stage changes for commit")
            display("git add html latex")
            if os.system('git add html latex'): raise OSError("Error trying to stage files for commit")
            print("\n# Commit changes locally")
            display('git commit -a -m "Automatic documentation generation at %s on %s"' % (gethostname(), datetime.now().strftime("%m-%d-%Y %H:%M")))
            if os.system('git commit -a -m "Automatic documentation generation at %s on %s"' % (gethostname(), datetime.now().strftime("%m-%d-%Y %H:%M"))):
                raise OSError("Error trying to commit files to local gh-pages branch")
            display("git push")
            print("\n# Push updated documentation upstream")
            display("git push")
            if os.system('git push'):
                raise OSError("While trying to push changes upstream 'git push' gave a nonzero return code")
            print("\n# Documentation successfully pushed upstream!")
        except:
            print_exc()
            input("\n# encountered ERROR (above). Will not push changes upstream.  Press a key")
        finally:
            print("\n# Switch back to master branch")
            display("git checkout master")
            os.system('git checkout master')
            for fnm in ["latex", "html"]:
                os.system("mv %s_ %s" % (fnm, fnm))

def add_tabs():
    """Adjust tabs in html version of documentation"""

    """
    Made obsolete in 2020-07-09
    for fnm in os.listdir('./html/'):
        if re.match('.*\.html$',fnm):
            fnm = './html/' + fnm
            newfile = []
            installtag = ' class="current"' if fnm.split('/')[-1] == 'installation.html' else ''
            usagetag = ' class="current"' if fnm.split('/')[-1] == 'usage.html' else ''
            tutorialtag = ' class="current"' if fnm.split('/')[-1] == 'tutorial.html' else ''
            glossarytag = ' class="current"' if fnm.split('/')[-1] == 'glossary.html' else ''
            todotag = ' class="current"' if fnm.split('/')[-1] == 'todo.html' else ''

            for line in open(fnm):
                newfile.append(line)
                if re.match('.*a href="index\.html"',line):
                    newfile.append('      <li%s><a href="installation.html"><span>Installation</span></a></li>\n' % installtag)
                    newfile.append('      <li%s><a href="usage.html"><span>Usage</span></a></li>\n' % usagetag)
                    newfile.append('      <li%s><a href="tutorial.html"><span>Tutorial</span></a></li>\n' % tutorialtag)
                    newfile.append('      <li%s><a href="glossary.html"><span>Glossary</span></a></li>\n' % glossarytag)
                    newfile.append('      <li><a href="api/roadmap.html"><span>API</span></a></li>\n')
            with open(fnm,'w') as f: f.writelines(newfile)
    """
    menudata_js = """
var menudata={children:[
{text:"Main Page",url:"index.html"},
{text:"Installation",url:"installation.html"},
{text:"Usage",url:"usage.html"},
{text:"Tutorial",url:"tutorial.html"},
{text:"Glossary",url:"glossary.html"},
{text:"Roadmap",url:"todo.html"},
{text:"API",url:"api/index.html"},
{text:"Files",url:"files.html",children:[
{text:"File List",url:"files.html"}]}]}
"""
    with open(os.path.join('html', 'menudata.js'), 'w') as f:
        f.write(menudata_js)

    for fnm in os.listdir('./html/api/'):
        if re.match('.*\.html$',fnm):
            fnm = './html/api/' + fnm
            newfile=[]
            for line in open(fnm):
                if re.match('.*a href="index\.html"',line):
                    newfile.append('      <li><a href="../index.html"><span>Main Page</span></a></li>\n')
                    newfile.append('      <li ')
                    if re.match('.*roadmap\.html$', fnm): newfile.append('class="current"')
                    newfile.append('><a href="roadmap.html"><span>Project Roadmap</span></a></li>\n')
                else: newfile.append(line)
            with open(fnm,'w') as f: f.writelines(newfile)

def find_forcebalance():
    """try to find forcebalance location in standard python path"""
    forcebalance_dir=""
    try:
        import forcebalance
        forcebalance_dir = forcebalance.__path__[0]
    except:
        print("Unable to find forcebalance directory in PYTHON PATH (Is it installed?)")
        print("Try running forcebalance/setup.py or you can always set the INPUT directory")
        print("manually in api.cfg")
        exit()

    print('ForceBalance directory is:', forcebalance_dir)
    return forcebalance_dir

def find_doxypy():
    """Check if doxypy is in system path or else ask for location of doxypy.py"""
    doxypy_path=""
    try:
        # first check to see if doxypy is in system path
        if subprocess.call(["doxypy", "makedocumentation.py"],stdout=open(os.devnull)): raise OSError()
        doxypy_path="doxypy"
    except OSError:
        doxypy_path=input("Enter location of doxypy.py: ")
        if not os.path.exists(doxypy_path) or doxypy_path[-9:] != 'doxypy.py':
            print("Invalid path to doxypy")
            exit()
    return doxypy_path

def build_config():
    forcebalance_path = find_forcebalance()
    doxypy_path = find_doxypy()

    with open('.doxygen.cfg', 'r') as f:
        lines = f.readlines()

    with open('doxygen.cfg', 'w') as f:
        for line in lines:
            if line.startswith('FILTER_PATTERNS        ='):
                option = 'FILTER_PATTERNS        = "*.py=' + doxypy_path + '"\n'
                f.write(option)
            else:
                f.write(line)

    with open('.api.cfg', 'r') as f:
        lines = f.readlines()

    with open('api.cfg', 'w') as f:
        for line in lines:
            if line.startswith('INPUT                  ='):
                option = 'INPUT                  = api.dox ' + forcebalance_path + '\n'
                f.write(option)
            elif line.startswith('FILTER_PATTERNS        ='):
                option = 'FILTER_PATTERNS        = "*.py=' + doxypy_path + '"\n'
                f.write(option)
            else:
                f.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', '-i', action='store_true', help="run in interactive mode, pausing before each command")
    parser.add_argument('--clean', '-k', action='store_true', help="remove temporary files after script is complete")
    parser.add_argument('--configure', '-c', action='store_true', help="generate doxygen configuration files from templates")
    parser.add_argument('--upstream', '-u', action='store_true', help="push updated documentation to upstream github repository")
    args = parser.parse_args()

    if args.configure:
        build_config()
    elif not os.path.isfile('doxygen.cfg') or not os.path.isfile('api.cfg'):
        print("Couldn't find required doxygen config files ('./doxygen.cfg' and './api.cfg').\nRun with --configure option to generate automatically")
        sys.exit(1)

    build(interactive = args.interactive, upstream = args.upstream)

    if args.clean:
        print("Cleaning up...")
        os.system("rm -rf latex option_index.txt api.dox mainpage.dox")   # cleanup
