"""
forcefield.py: Constructs OpenMM System objects based on a Topology and an XML force field description

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2012-2014 Stanford University and the Authors.
Authors: Peter Eastman, Mark Friedrichs
Contributors:

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
__author__ = "Peter Eastman"
__version__ = "1.0"

import os
import itertools
import xml.etree.ElementTree as etree
from math import sqrt, cos
import simtk.openmm as mm
import simtk.unit as unit
import simtk.openmm.app.element as elem
from simtk.openmm.app import Topology

def _convertParameterToNumber(param):
    if unit.is_quantity(param):
        return mm.stripUnits((param,))[0]
    return float(param)

# Enumerated values for nonbonded method

class NoCutoff(object):
    def __repr__(self):
        return 'NoCutoff'
NoCutoff = NoCutoff()

class CutoffNonPeriodic(object):
    def __repr__(self):
        return 'CutoffNonPeriodic'
CutoffNonPeriodic = CutoffNonPeriodic()

class CutoffPeriodic(object):
    def __repr__(self):
        return 'CutoffPeriodic'
CutoffPeriodic = CutoffPeriodic()

class Ewald(object):
    def __repr__(self):
        return 'Ewald'
Ewald = Ewald()

class PME(object):
    def __repr__(self):
        return 'PME'
PME = PME()

# Enumerated values for constraint type

class HBonds(object):
    def __repr__(self):
        return 'HBonds'
HBonds = HBonds()

class AllBonds(object):
    def __repr__(self):
        return 'AllBonds'
AllBonds = AllBonds()

class HAngles(object):
    def __repr__(self):
        return 'HAngles'
HAngles = HAngles()

# A map of functions to parse elements of the XML file.

parsers = {}

class ForceField(object):
    """A ForceField constructs OpenMM System objects based on a Topology."""

    def __init__(self, *files):
        """Load one or more XML files and create a ForceField object based on them.

        Parameters:
         - files (list) A list of XML files defining the force field.  Each entry may
           be an absolute file path, a path relative to the current working
           directory, a path relative to this module's data subdirectory
           (for built in force fields), or an open file-like object with a
           read() method from which the forcefield XML data can be loaded.
        """
        self._atomTypes = {}
        self._templates = {}
        self._templateSignatures = {None:[]}
        self._atomClasses = {'':set()}
        self._forces = []
        self._scripts = []
        for file in files:
            self.loadFile(file)
    
    def loadFile(self, file):
        """Load an XML file and add the definitions from it to this FieldField.
        
        Parameters:
         - file (string or file) An XML file containing force field definitions.  It may
           be either an absolute file path, a path relative to the current working
           directory, a path relative to this module's data subdirectory
           (for built in force fields), or an open file-like object with a
           read() method from which the forcefield XML data can be loaded.
        """
        try:
            # this handles either filenames or open file-like objects
            tree = etree.parse(file)
        except IOError:
            tree = etree.parse(os.path.join(os.path.dirname(__file__), 'data', file))
        root = tree.getroot()

        # Load the atom types.

        if tree.getroot().find('AtomTypes') is not None:
            for type in tree.getroot().find('AtomTypes').findall('Type'):
                self.registerAtomType(type.attrib)

        # Load the residue templates.

        if tree.getroot().find('Residues') is not None:
            for residue in root.find('Residues').findall('Residue'):
                resName = residue.attrib['name']
                template = ForceField._TemplateData(resName)
                for atom in residue.findall('Atom'):
                    template.atoms.append(ForceField._TemplateAtomData(atom.attrib['name'], atom.attrib['type'], self._atomTypes[atom.attrib['type']][2]))
                for site in residue.findall('VirtualSite'):
                    template.virtualSites.append(ForceField._VirtualSiteData(site))
                for bond in residue.findall('Bond'):
                    template.addBond(int(bond.attrib['from']), int(bond.attrib['to']))
                for bond in residue.findall('ExternalBond'):
                    b = int(bond.attrib['from'])
                    template.externalBonds.append(b)
                    template.atoms[b].externalBonds += 1
                self.registerResidueTemplate(template)

        # Load force definitions

        for child in root:
            if child.tag in parsers:
                parsers[child.tag](child, self)

        # Load scripts

        for node in tree.getroot().findall('Script'):
            self.registerScript(node.text)

    def getGenerators(self):
        """Get the list of all registered generators."""
        return self._forces
    
    def registerGenerator(self, generator):
        """Register a new generator."""
        self._forces.append(generator)
    
    def registerAtomType(self, parameters):
        """Register a new atom type."""
        name = parameters['name']
        if name in self._atomTypes:
            raise ValueError('Found multiple definitions for atom type: '+name)
        atomClass = parameters['class']
        mass = _convertParameterToNumber(parameters['mass'])
        element = None
        if 'element' in parameters:
            element = parameters['element']
            if not isinstance(element, elem.Element):
                element = elem.get_by_symbol(element)
        self._atomTypes[name] = (atomClass, mass, element)
        if atomClass in self._atomClasses:
            typeSet = self._atomClasses[atomClass]
        else:
            typeSet = set()
            self._atomClasses[atomClass] = typeSet
        typeSet.add(name)
        self._atomClasses[''].add(name)
    
    def registerResidueTemplate(self, template):
        """Register a new residue template."""
        self._templates[template.name] = template
        signature = _createResidueSignature([atom.element for atom in template.atoms])
        if signature in self._templateSignatures:
            self._templateSignatures[signature].append(template)
        else:
            self._templateSignatures[signature] = [template]
    
    def registerScript(self, script):
        """Register a new script to be executed after building the System."""
        self._scripts.append(script)

    def _findAtomTypes(self, attrib, num):
        """Parse the attributes on an XML tag to find the set of atom types for each atom it involves."""
        types = []
        for i in range(num):
            if num == 1:
                suffix = ''
            else:
                suffix = str(i+1)
            classAttrib = 'class'+suffix
            typeAttrib = 'type'+suffix
            if classAttrib in attrib:
                if typeAttrib in attrib:
                    raise ValueError('Specified both a type and a class for the same atom: '+str(attrib))
                if attrib[classAttrib] not in self._atomClasses:
                    types.append(None) # Unknown atom class
                else:
                    types.append(self._atomClasses[attrib[classAttrib]])
            else:
                if typeAttrib not in attrib or attrib[typeAttrib] not in self._atomTypes:
                    types.append(None) # Unknown atom type
                else:
                    types.append([attrib[typeAttrib]])
        return types

    def _parseTorsion(self, attrib):
        """Parse the node defining a torsion."""
        types = self._findAtomTypes(attrib, 4)
        if None in types:
            return None
        torsion = PeriodicTorsion(types)
        index = 1
        while 'phase%d'%index in attrib:
            torsion.periodicity.append(int(attrib['periodicity%d'%index]))
            torsion.phase.append(_convertParameterToNumber(attrib['phase%d'%index]))
            torsion.k.append(_convertParameterToNumber(attrib['k%d'%index]))
            index += 1
        return torsion

    class _SystemData:
        """Inner class used to encapsulate data about the system being created."""
        def __init__(self):
            self.atomType = {}
            self.atoms = []
            self.excludeAtomWith = []
            self.virtualSites = {}
            self.bonds = []
            self.angles = []
            self.propers = []
            self.impropers = []
            self.atomBonds = []
            self.isAngleConstrained = []

    class _TemplateData:
        """Inner class used to encapsulate data about a residue template definition."""
        def __init__(self, name):
            self.name = name
            self.atoms = []
            self.virtualSites = []
            self.bonds = []
            self.externalBonds = []
        
        def addBond(self, atom1, atom2):
            self.bonds.append((atom1, atom2))
            self.atoms[atom1].bondedTo.append(atom2)
            self.atoms[atom2].bondedTo.append(atom1)

    class _TemplateAtomData:
        """Inner class used to encapsulate data about an atom in a residue template definition."""
        def __init__(self, name, type, element):
            self.name = name
            self.type = type
            self.element = element
            self.bondedTo = []
            self.externalBonds = 0

    class _BondData:
        """Inner class used to encapsulate data about a bond."""
        def __init__(self, atom1, atom2):
            self.atom1 = atom1
            self.atom2 = atom2
            self.isConstrained = False
            self.length = 0.0

    class _VirtualSiteData:
        """Inner class used to encapsulate data about a virtual site."""
        def __init__(self, node):
            attrib = node.attrib
            self.index = int(attrib['index'])
            self.type = attrib['type']
            if self.type == 'average2':
                self.atoms = [int(attrib['atom1']), int(attrib['atom2'])]
                self.weights = [float(attrib['weight1']), float(attrib['weight2'])]
            elif self.type == 'average3':
                self.atoms = [int(attrib['atom1']), int(attrib['atom2']), int(attrib['atom3'])]
                self.weights = [float(attrib['weight1']), float(attrib['weight2']), float(attrib['weight3'])]
            elif self.type == 'outOfPlane':
                self.atoms = [int(attrib['atom1']), int(attrib['atom2']), int(attrib['atom3'])]
                self.weights = [float(attrib['weight12']), float(attrib['weight13']), float(attrib['weightCross'])]
            elif self.type == 'localCoords':
                self.atoms = [int(attrib['atom1']), int(attrib['atom2']), int(attrib['atom3'])]
                self.originWeights = [float(attrib['wo1']), float(attrib['wo2']), float(attrib['wo3'])]
                self.xWeights = [float(attrib['wx1']), float(attrib['wx2']), float(attrib['wx3'])]
                self.yWeights = [float(attrib['wy1']), float(attrib['wy2']), float(attrib['wy3'])]
                self.localPos = [float(attrib['p1']), float(attrib['p2']), float(attrib['p3'])]
            else:
                raise ValueError('Unknown virtual site type: %s' % self.type)
            if 'excludeWith' in attrib:
                self.excludeWith = int(attrib['excludeWith'])
            else:
                self.excludeWith = self.atoms[0]

    def createSystem(self, topology, nonbondedMethod=NoCutoff, nonbondedCutoff=1.0*unit.nanometer,
                     constraints=None, rigidWater=True, removeCMMotion=True, hydrogenMass=None, **args):
        """Construct an OpenMM System representing a Topology with this force field.

        Parameters:
         - topology (Topology) The Topology for which to create a System
         - nonbondedMethod (object=NoCutoff) The method to use for nonbonded interactions.  Allowed values are
           NoCutoff, CutoffNonPeriodic, CutoffPeriodic, Ewald, or PME.
         - nonbondedCutoff (distance=1*nanometer) The cutoff distance to use for nonbonded interactions
         - constraints (object=None) Specifies which bonds and angles should be implemented with constraints.
           Allowed values are None, HBonds, AllBonds, or HAngles.
         - rigidWater (boolean=True) If true, water molecules will be fully rigid regardless of the value passed for the constraints argument
         - removeCMMotion (boolean=True) If true, a CMMotionRemover will be added to the System
         - hydrogenMass (mass=None) The mass to use for hydrogen atoms bound to heavy atoms.  Any mass added to a hydrogen is
           subtracted from the heavy atom to keep their total mass the same.
         - args Arbitrary additional keyword arguments may also be specified.  This allows extra parameters to be specified that are specific to
           particular force fields.
        Returns: the newly created System
        """
        data = ForceField._SystemData()
        data.atoms = list(topology.atoms())
        for atom in data.atoms:
            data.excludeAtomWith.append([])

        # Make a list of all bonds

        for bond in topology.bonds():
            data.bonds.append(ForceField._BondData(bond[0].index, bond[1].index))

        # Record which atoms are bonded to each other atom

        bondedToAtom = []
        for i in range(len(data.atoms)):
            bondedToAtom.append(set())
            data.atomBonds.append([])
        for i in range(len(data.bonds)):
            bond = data.bonds[i]
            bondedToAtom[bond.atom1].add(bond.atom2)
            bondedToAtom[bond.atom2].add(bond.atom1)
            data.atomBonds[bond.atom1].append(i)
            data.atomBonds[bond.atom2].append(i)

        # Find the template matching each residue and assign atom types.

        for chain in topology.chains():
            for res in chain.residues():
                template = None
                matches = None
                signature = _createResidueSignature([atom.element for atom in res.atoms()])
                if signature in self._templateSignatures:
                    for t in self._templateSignatures[signature]:
                        matches = _matchResidue(res, t, bondedToAtom)
                        if matches is not None:
                            template = t
                            break
                if matches is None:
                    raise ValueError('No template found for residue %d (%s).  %s' % (res.index+1, res.name, _findMatchErrors(self, res)))
                matchAtoms = dict(zip(matches, res.atoms()))
                for atom, match in zip(res.atoms(), matches):
                    data.atomType[atom] = template.atoms[match].type
                    for site in template.virtualSites:
                        if match == site.index:
                            data.virtualSites[atom] = (site, [matchAtoms[i].index for i in site.atoms], matchAtoms[site.excludeWith].index)

        # Create the System and add atoms

        sys = mm.System()
        for atom in topology.atoms():
            sys.addParticle(self._atomTypes[data.atomType[atom]][1])
        
        # Adjust masses.
        
        if hydrogenMass is not None:
            for atom1, atom2 in topology.bonds():
                if atom1.element == elem.hydrogen:
                    (atom1, atom2) = (atom2, atom1)
                if atom2.element == elem.hydrogen and atom1.element not in (elem.hydrogen, None):
                    transferMass = hydrogenMass-sys.getParticleMass(atom2.index)
                    sys.setParticleMass(atom2.index, hydrogenMass)
                    sys.setParticleMass(atom1.index, sys.getParticleMass(atom1.index)-transferMass)

        # Set periodic boundary conditions.

        boxSize = topology.getUnitCellDimensions()
        if boxSize is not None:
            sys.setDefaultPeriodicBoxVectors((boxSize[0], 0, 0), (0, boxSize[1], 0), (0, 0, boxSize[2]))
        elif nonbondedMethod is not NoCutoff and nonbondedMethod is not CutoffNonPeriodic:
            raise ValueError('Requested periodic boundary conditions for a Topology that does not specify periodic box dimensions')

        # Make a list of all unique angles

        uniqueAngles = set()
        for bond in data.bonds:
            for atom in bondedToAtom[bond.atom1]:
                if atom != bond.atom2:
                    if atom < bond.atom2:
                        uniqueAngles.add((atom, bond.atom1, bond.atom2))
                    else:
                        uniqueAngles.add((bond.atom2, bond.atom1, atom))
            for atom in bondedToAtom[bond.atom2]:
                if atom != bond.atom1:
                    if atom > bond.atom1:
                        uniqueAngles.add((bond.atom1, bond.atom2, atom))
                    else:
                        uniqueAngles.add((atom, bond.atom2, bond.atom1))
        data.angles = sorted(list(uniqueAngles))

        # Make a list of all unique proper torsions

        uniquePropers = set()
        for angle in data.angles:
            for atom in bondedToAtom[angle[0]]:
                if atom != angle[1]:
                    if atom < angle[2]:
                        uniquePropers.add((atom, angle[0], angle[1], angle[2]))
                    else:
                        uniquePropers.add((angle[2], angle[1], angle[0], atom))
            for atom in bondedToAtom[angle[2]]:
                if atom != angle[1]:
                    if atom > angle[0]:
                        uniquePropers.add((angle[0], angle[1], angle[2], atom))
                    else:
                        uniquePropers.add((atom, angle[2], angle[1], angle[0]))
        data.propers = sorted(list(uniquePropers))

        # Make a list of all unique improper torsions

        for atom in range(len(bondedToAtom)):
            bondedTo = bondedToAtom[atom]
            if len(bondedTo) > 2:
                for subset in itertools.combinations(bondedTo, 3):
                    data.impropers.append((atom, subset[0], subset[1], subset[2]))

        # Identify bonds that should be implemented with constraints

        if constraints == AllBonds or constraints == HAngles:
            for bond in data.bonds:
                bond.isConstrained = True
        elif constraints == HBonds:
            for bond in data.bonds:
                atom1 = data.atoms[bond.atom1]
                atom2 = data.atoms[bond.atom2]
                bond.isConstrained = atom1.name.startswith('H') or atom2.name.startswith('H')
        if rigidWater:
            for bond in data.bonds:
                atom1 = data.atoms[bond.atom1]
                atom2 = data.atoms[bond.atom2]
                if atom1.residue.name == 'HOH' and atom2.residue.name == 'HOH':
                    bond.isConstrained = True

        # Identify angles that should be implemented with constraints

        if constraints == HAngles:
            for angle in data.angles:
                atom1 = data.atoms[angle[0]]
                atom2 = data.atoms[angle[1]]
                atom3 = data.atoms[angle[2]]
                numH = 0
                if atom1.name.startswith('H'):
                    numH += 1
                if atom3.name.startswith('H'):
                    numH += 1
                data.isAngleConstrained.append(numH == 2 or (numH == 1 and atom2.name.startswith('O')))
        else:
            data.isAngleConstrained = len(data.angles)*[False]
        if rigidWater:
            for i in range(len(data.angles)):
                angle = data.angles[i]
                atom1 = data.atoms[angle[0]]
                atom2 = data.atoms[angle[1]]
                atom3 = data.atoms[angle[2]]
                if atom1.residue.name == 'HOH' and atom2.residue.name == 'HOH' and atom3.residue.name == 'HOH':
                    data.isAngleConstrained[i] = True

        # Add virtual sites

        for atom in data.virtualSites:
            (site, atoms, excludeWith) = data.virtualSites[atom]
            index = atom.index
            data.excludeAtomWith[excludeWith].append(index)
            if site.type == 'average2':
                sys.setVirtualSite(index, mm.TwoParticleAverageSite(atoms[0], atoms[1], site.weights[0], site.weights[1]))
            elif site.type == 'average3':
                sys.setVirtualSite(index, mm.ThreeParticleAverageSite(atoms[0], atoms[1], atoms[2], site.weights[0], site.weights[1], site.weights[2]))
            elif site.type == 'outOfPlane':
                sys.setVirtualSite(index, mm.OutOfPlaneSite(atoms[0], atoms[1], atoms[2], site.weights[0], site.weights[1], site.weights[2]))
            elif site.type == 'localCoords':
                sys.setVirtualSite(index, mm.LocalCoordinatesSite(atoms[0], atoms[1], atoms[2],
                                                                  mm.Vec3(site.originWeights[0], site.originWeights[1], site.originWeights[2]),
                                                                  mm.Vec3(site.xWeights[0], site.xWeights[1], site.xWeights[2]),
                                                                  mm.Vec3(site.yWeights[0], site.yWeights[1], site.yWeights[2]),
                                                                  mm.Vec3(site.localPos[0], site.localPos[1], site.localPos[2])))

        # Add forces to the System

        for force in self._forces:
            force.createForce(sys, data, nonbondedMethod, nonbondedCutoff, args)
        if removeCMMotion:
            sys.addForce(mm.CMMotionRemover())

        # Let generators do postprocessing

        for force in self._forces:
            if 'postprocessSystem' in dir(force):
                force.postprocessSystem(sys, data, args)

        # Execute scripts found in the XML files.

        for script in self._scripts:
            exec script
        return sys


def _countResidueAtoms(elements):
    """Count the number of atoms of each element in a residue."""
    counts = {}
    for element in elements:
        if element in counts:
            counts[element] += 1
        else:
            counts[element] = 1
    return counts


def _createResidueSignature(elements):
    """Create a signature for a residue based on the elements of the atoms it contains."""
    counts = _countResidueAtoms(elements)
    sig = []
    for c in counts:
        if c is not None:
            sig.append((c, counts[c]))
    sig.sort(key=lambda x: -x[0].mass)

    # Convert it to a string.

    s = ''
    for element, count in sig:
        s += element.symbol+str(count)
    return s


def _matchResidue(res, template, bondedToAtom):
    """Determine whether a residue matches a template and return a list of corresponding atoms.

    Parameters:
     - res (Residue) The residue to check
     - template (_TemplateData) The template to compare it to
     - bondedToAtom (list) Enumerates which other atoms each atom is bonded to
    Returns: a list specifying which atom of the template each atom of the residue corresponds to,
    or None if it does not match the template
    """
    atoms = list(res.atoms())
    if len(atoms) != len(template.atoms):
        return None
    matches = len(atoms)*[0]
    hasMatch = len(atoms)*[False]

    # Translate from global to local atom indices, and record the bonds for each atom.

    renumberAtoms = {}
    for i in range(len(atoms)):
        renumberAtoms[atoms[i].index] = i
    bondedTo = []
    externalBonds = []
    for atom in atoms:
        bonds = [renumberAtoms[x] for x in bondedToAtom[atom.index] if x in renumberAtoms]
        bondedTo.append(bonds)
        externalBonds.append(len([x for x in bondedToAtom[atom.index] if x not in renumberAtoms]))

    # For each unique combination of element and number of bonds, make sure the residue and
    # template have the same number of atoms.

    residueTypeCount = {}
    for i, atom in enumerate(atoms):
        key = (atom.element, len(bondedTo[i]), externalBonds[i])
        if key not in residueTypeCount:
            residueTypeCount[key] = 1
        residueTypeCount[key] += 1
    templateTypeCount = {}
    for i, atom in enumerate(template.atoms):
        key = (atom.element, len(atom.bondedTo), atom.externalBonds)
        if key not in templateTypeCount:
            templateTypeCount[key] = 1
        templateTypeCount[key] += 1
    if residueTypeCount != templateTypeCount:
        return None

    # Recursively match atoms.

    if _findAtomMatches(atoms, template, bondedTo, externalBonds, matches, hasMatch, 0):
        return matches
    return None


def _findAtomMatches(atoms, template, bondedTo, externalBonds, matches, hasMatch, position):
    """This is called recursively from inside _matchResidue() to identify matching atoms."""
    if position == len(atoms):
        return True
    elem = atoms[position].element
    name = atoms[position].name
    for i in range(len(atoms)):
        atom = template.atoms[i]
        if ((atom.element is not None and atom.element == elem) or (atom.element is None and atom.name == name)) and not hasMatch[i] and len(atom.bondedTo) == len(bondedTo[position]) and atom.externalBonds == externalBonds[position]:
            # See if the bonds for this identification are consistent

            allBondsMatch = all((bonded > position or matches[bonded] in atom.bondedTo for bonded in bondedTo[position]))
            if allBondsMatch:
                # This is a possible match, so trying matching the rest of the residue.

                matches[position] = i
                hasMatch[i] = True
                if _findAtomMatches(atoms, template, bondedTo, externalBonds, matches, hasMatch, position+1):
                    return True
                hasMatch[i] = False
    return False


def _findMatchErrors(forcefield, res):
    """Try to guess why a residue failed to match any template and return an error message."""
    residueCounts = _countResidueAtoms([atom.element for atom in res.atoms()])
    numResidueAtoms = sum(residueCounts.itervalues())
    numResidueHeavyAtoms = sum(residueCounts[element] for element in residueCounts if element not in (None, elem.hydrogen))
    
    # Loop over templates and see how closely each one might match.
    
    bestMatchName = None
    numBestMatchAtoms = 3*numResidueAtoms
    numBestMatchHeavyAtoms = 2*numResidueHeavyAtoms
    for templateName in forcefield._templates:
        template = forcefield._templates[templateName]
        templateCounts = _countResidueAtoms([atom.element for atom in template.atoms])
        
        # Does the residue have any atoms that clearly aren't in the template?
        
        if any(element not in templateCounts or templateCounts[element] < residueCounts[element] for element in residueCounts):
            continue
        
        # If there are too many missing atoms, discard this template.
        
        numTemplateAtoms = sum(templateCounts.itervalues())
        numTemplateHeavyAtoms = sum(templateCounts[element] for element in templateCounts if element not in (None, elem.hydrogen))
        if numTemplateAtoms > numBestMatchAtoms:
            continue
        if numTemplateHeavyAtoms > numBestMatchHeavyAtoms:
            continue
        
        # If this template has the same number of missing atoms as our previous best one, look at the name
        # to decide which one to use.
        
        if numTemplateAtoms == numBestMatchAtoms:
            if bestMatchName == res.name or res.name not in templateName:
                continue
        
        # Accept this as our new best match.
        
        bestMatchName = templateName
        numBestMatchAtoms = numTemplateAtoms
        numBestMatchHeavyAtoms = numTemplateHeavyAtoms
        numBestMatchExtraParticles = len([atom for atom in template.atoms if atom.element is None])
    
    # Return an appropriate error message.
    
    if numBestMatchAtoms == numResidueAtoms:
        chainResidues = list(res.chain.residues())
        if len(chainResidues) > 1 and (res == chainResidues[0] or res == chainResidues[-1]):
            return 'The set of atoms matches %s, but the bonds are different.  Perhaps the chain is missing a terminal group?' % bestMatchName
        return 'The set of atoms matches %s, but the bonds are different.' % bestMatchName
    if bestMatchName is not None:
        if numBestMatchHeavyAtoms == numResidueHeavyAtoms:
            numResidueExtraParticles = len([atom for atom in res.atoms() if atom.element is None])
            if numResidueExtraParticles == 0 and numBestMatchExtraParticles == 0:
                return 'The set of atoms is similar to %s, but it is missing %d hydrogen atoms.' % (bestMatchName, numBestMatchAtoms-numResidueAtoms)
            if numBestMatchExtraParticles-numResidueExtraParticles == numBestMatchAtoms-numResidueAtoms:
                return 'The set of atoms is similar to %s, but it is missing %d extra particles.  You can add them with Modeller.addExtraParticles().' % (bestMatchName, numBestMatchAtoms-numResidueAtoms)
        return 'The set of atoms is similar to %s, but it is missing %d atoms.' % (bestMatchName, numBestMatchAtoms-numResidueAtoms)
    return 'This might mean your input topology is missing some atoms or bonds, or possibly that you are using the wrong force field.'


# The following classes are generators that know how to create Force subclasses and add them to a System that is being
# created.  Each generator class must define two methods: 1) a static method that takes an etree Element and a ForceField,
# and returns the corresponding generator object; 2) a createForce() method that constructs the Force object and adds it
# to the System.  The static method should be added to the parsers map.

class HarmonicBondTerm(object):
    typedict = {}
    my_bonds = []
    my_constraints = []

    def __init__(self, type0, type1, length, k):
        self.type0 = type0
        self.type1 = type1
        self.length = length
        self.k = k
        
        if (type0, type1) not in self.typedict:
            self.typedict[type0, type1] = []
        if (type1, type0) not in self.typedict:
            self.typedict[type1, type0] = []

        self.typedict[type0, type1].append(self)
        self.typedict[type1, type0].append(self)

    
    def add_force(self, sys, force, bond):
        if bond.isConstrained:
            sys.addConstraint(bond.atom1, bond.atom2, self.length)
            self.my_constraints.append((bond.atom1, bond.atom2))
        elif self.k != 0:
            force.addBond(bond.atom1, bond.atom2, self.length, self.k)
            self.my_bonds.append((bond.atom1, bond.atom2))
    
    @classmethod
    def lookup(cls, type0, type1):
        print(type0, type1)
        print(cls.typedict)
        try:
            return cls.typedict[type0, type1]
        except KeyError:
            return []

## @private
class HarmonicBondGenerator:
    """A HarmonicBondGenerator constructs a HarmonicBondForce."""

    def __init__(self, forcefield):
        self.ff = forcefield
    
    def registerBond(self, parameters):
        types = self.ff._findAtomTypes(parameters, 2)
        if None not in types:
            _unused = HarmonicBondTerm(list(types[0])[0], list(types[1])[0], _convertParameterToNumber(parameters['length']), _convertParameterToNumber(parameters['k']))

    @staticmethod
    def parseElement(element, ff):
        generator = HarmonicBondGenerator(ff)
        ff.registerGenerator(generator)
        for bond in element.findall('Bond'):
            generator.registerBond(bond.attrib)

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        existing = [sys.getForce(i) for i in range(sys.getNumForces())]
        existing = [f for f in existing if type(f) == mm.HarmonicBondForce]
        if len(existing) == 0:
            force = mm.HarmonicBondForce()
            sys.addForce(force)
        else:
            force = existing[0]
        for bond in data.bonds:
            print(bond)
            type1 = data.atomType[data.atoms[bond.atom1]]
            type2 = data.atomType[data.atoms[bond.atom2]]
            
            bond_terms = HarmonicBondTerm.lookup(type1, type2)
            print(bond_terms)
            for bond_term in bond_terms:
                print(bond_term)
                bond_term.add_force(sys, force, bond)
            

parsers["HarmonicBondForce"] = HarmonicBondGenerator.parseElement


## @private
class HarmonicAngleGenerator:
    """A HarmonicAngleGenerator constructs a HarmonicAngleForce."""

    def __init__(self, forcefield):
        self.ff = forcefield
        self.types1 = []
        self.types2 = []
        self.types3 = []
        self.angle = []
        self.k = []

    def registerAngle(self, parameters):
        types = self.ff._findAtomTypes(parameters, 3)
        if None not in types:
            self.types1.append(types[0])
            self.types2.append(types[1])
            self.types3.append(types[2])
            self.angle.append(_convertParameterToNumber(parameters['angle']))
            self.k.append(_convertParameterToNumber(parameters['k']))

    @staticmethod
    def parseElement(element, ff):
        generator = HarmonicAngleGenerator(ff)
        ff.registerGenerator(generator)
        for angle in element.findall('Angle'):
            generator.registerAngle(angle.attrib)

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        existing = [sys.getForce(i) for i in range(sys.getNumForces())]
        existing = [f for f in existing if type(f) == mm.HarmonicAngleForce]
        if len(existing) == 0:
            force = mm.HarmonicAngleForce()
            sys.addForce(force)
        else:
            force = existing[0]
        for (angle, isConstrained) in zip(data.angles, data.isAngleConstrained):
            type1 = data.atomType[data.atoms[angle[0]]]
            type2 = data.atomType[data.atoms[angle[1]]]
            type3 = data.atomType[data.atoms[angle[2]]]
            for i in range(len(self.types1)):
                types1 = self.types1[i]
                types2 = self.types2[i]
                types3 = self.types3[i]
                if (type1 in types1 and type2 in types2 and type3 in types3) or (type1 in types3 and type2 in types2 and type3 in types1):
                    if isConstrained:
                        # Find the two bonds that make this angle.

                        bond1 = None
                        bond2 = None
                        for bond in data.atomBonds[angle[1]]:
                            atom1 = data.bonds[bond].atom1
                            atom2 = data.bonds[bond].atom2
                            if atom1 == angle[0] or atom2 == angle[0]:
                                bond1 = bond
                            elif atom1 == angle[2] or atom2 == angle[2]:
                                bond2 = bond

                        # Compute the distance between atoms and add a constraint

                        if bond1 is not None and bond2 is not None:
                            l1 = data.bonds[bond1].length
                            l2 = data.bonds[bond2].length
                            if l1 is not None and l2 is not None:
                                length = sqrt(l1*l1 + l2*l2 - 2*l1*l2*cos(self.angle[i]))
                                sys.addConstraint(angle[0], angle[2], length)
                    elif self.k[i] != 0:
                        force.addAngle(angle[0], angle[1], angle[2], self.angle[i], self.k[i])
                    break

parsers["HarmonicAngleForce"] = HarmonicAngleGenerator.parseElement


## @private
class PeriodicTorsion:
    """A PeriodicTorsion records the information for a periodic torsion definition."""

    def __init__(self, types):
        self.types1 = types[0]
        self.types2 = types[1]
        self.types3 = types[2]
        self.types4 = types[3]
        self.periodicity = []
        self.phase = []
        self.k = []

## @private
class PeriodicTorsionGenerator:
    """A PeriodicTorsionGenerator constructs a PeriodicTorsionForce."""

    def __init__(self, forcefield):
        self.ff = forcefield
        self.proper = []
        self.improper = []

    def registerProperTorsion(self, parameters):
        torsion = self.ff._parseTorsion(parameters)
        if torsion is not None:
            self.proper.append(torsion)

    def registerImproperTorsion(self, parameters):
        torsion = self.ff._parseTorsion(parameters)
        if torsion is not None:
            self.improper.append(torsion)

    @staticmethod
    def parseElement(element, ff):
        generator = PeriodicTorsionGenerator(ff)
        ff.registerGenerator(generator)
        for torsion in element.findall('Proper'):
            generator.registerProperTorsion(torsion.attrib)
        for torsion in element.findall('Improper'):
            generator.registerImproperTorsion(torsion.attrib)

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        existing = [sys.getForce(i) for i in range(sys.getNumForces())]
        existing = [f for f in existing if type(f) == mm.PeriodicTorsionForce]
        if len(existing) == 0:
            force = mm.PeriodicTorsionForce()
            sys.addForce(force)
        else:
            force = existing[0]
        wildcard = self.ff._atomClasses['']
        for torsion in data.propers:
            type1 = data.atomType[data.atoms[torsion[0]]]
            type2 = data.atomType[data.atoms[torsion[1]]]
            type3 = data.atomType[data.atoms[torsion[2]]]
            type4 = data.atomType[data.atoms[torsion[3]]]
            match = None
            for tordef in self.proper:
                types1 = tordef.types1
                types2 = tordef.types2
                types3 = tordef.types3
                types4 = tordef.types4
                if (type2 in types2 and type3 in types3 and type4 in types4 and type1 in types1) or (type2 in types3 and type3 in types2 and type4 in types1 and type1 in types4):
                    hasWildcard = (wildcard in (types1, types2, types3, types4))
                    if match is None or not hasWildcard: # Prefer specific definitions over ones with wildcards
                        match = tordef
                    if not hasWildcard:
                        break
            if match is not None:
                for i in range(len(match.phase)):
                    if match.k[i] != 0:
                        force.addTorsion(torsion[0], torsion[1], torsion[2], torsion[3], match.periodicity[i], match.phase[i], match.k[i])
        for torsion in data.impropers:
            type1 = data.atomType[data.atoms[torsion[0]]]
            type2 = data.atomType[data.atoms[torsion[1]]]
            type3 = data.atomType[data.atoms[torsion[2]]]
            type4 = data.atomType[data.atoms[torsion[3]]]
            done = False
            for tordef in self.improper:
                if done:
                    break
                types1 = tordef.types1
                types2 = tordef.types2
                types3 = tordef.types3
                types4 = tordef.types4
                if type1 in types1:
                    for (t2, t3, t4) in itertools.permutations(((type2, 1), (type3, 2), (type4, 3))):
                        if t2[0] in types2 and t3[0] in types3 and t4[0] in types4:
                            # Workaround to be more consistent with AMBER.  It uses wildcards to define most of its
                            # impropers, which leaves the ordering ambiguous.  It then follows some bizarre rules
                            # to pick the order.
                            a1 = torsion[t2[1]]
                            a2 = torsion[t3[1]]
                            e1 = data.atoms[a1].element
                            e2 = data.atoms[a2].element
                            if e1 == e2 and a1 > a2:
                                (a1, a2) = (a2, a1)
                            elif e1 != elem.carbon and (e2 == elem.carbon or e1.mass < e2.mass):
                                (a1, a2) = (a2, a1)
                            for i in range(len(tordef.phase)):
                                if tordef.k[i] != 0:
                                    force.addTorsion(a1, a2, torsion[0], torsion[t4[1]], tordef.periodicity[i], tordef.phase[i], tordef.k[i])
                            done = True
                            break

parsers["PeriodicTorsionForce"] = PeriodicTorsionGenerator.parseElement


## @private
class RBTorsion:
    """An RBTorsion records the information for a Ryckaert-Bellemans torsion definition."""

    def __init__(self, types, c):
        self.types1 = types[0]
        self.types2 = types[1]
        self.types3 = types[2]
        self.types4 = types[3]
        self.c = c

## @private
class RBTorsionGenerator:
    """An RBTorsionGenerator constructs an RBTorsionForce."""

    def __init__(self, forcefield):
        self.ff = forcefield
        self.proper = []
        self.improper = []

    @staticmethod
    def parseElement(element, ff):
        generator = RBTorsionGenerator(ff)
        ff.registerGenerator(generator)
        for torsion in element.findall('Proper'):
            types = ff._findAtomTypes(torsion.attrib, 4)
            if None not in types:
                generator.proper.append(RBTorsion(types, [float(torsion.attrib['c'+str(i)]) for i in range(6)]))
        for torsion in element.findall('Improper'):
            types = ff._findAtomTypes(torsion.attrib, 4)
            if None not in types:
                generator.improper.append(RBTorsion(types, [float(torsion.attrib['c'+str(i)]) for i in range(6)]))

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        existing = [sys.getForce(i) for i in range(sys.getNumForces())]
        existing = [f for f in existing if type(f) == mm.RBTorsionForce]
        if len(existing) == 0:
            force = mm.RBTorsionForce()
            sys.addForce(force)
        else:
            force = existing[0]
        wildcard = self.ff._atomClasses['']
        for torsion in data.propers:
            type1 = data.atomType[data.atoms[torsion[0]]]
            type2 = data.atomType[data.atoms[torsion[1]]]
            type3 = data.atomType[data.atoms[torsion[2]]]
            type4 = data.atomType[data.atoms[torsion[3]]]
            match = None
            for tordef in self.proper:
                types1 = tordef.types1
                types2 = tordef.types2
                types3 = tordef.types3
                types4 = tordef.types4
                if (type2 in types2 and type3 in types3 and type4 in types4 and type1 in types1) or (type2 in types3 and type3 in types2 and type4 in types1 and type1 in types4):
                    hasWildcard = (wildcard in (types1, types2, types3, types4))
                    if match is None or not hasWildcard: # Prefer specific definitions over ones with wildcards
                        match = tordef
                    if not hasWildcard:
                        break
            if match is not None:
                force.addTorsion(torsion[0], torsion[1], torsion[2], torsion[3], match.c[0], match.c[1], match.c[2], match.c[3], match.c[4], match.c[5])
        for torsion in data.impropers:
            type1 = data.atomType[data.atoms[torsion[0]]]
            type2 = data.atomType[data.atoms[torsion[1]]]
            type3 = data.atomType[data.atoms[torsion[2]]]
            type4 = data.atomType[data.atoms[torsion[3]]]
            done = False
            for tordef in self.improper:
                if done:
                    break
                types1 = tordef.types1
                types2 = tordef.types2
                types3 = tordef.types3
                types4 = tordef.types4
                if type1 in types1:
                    for (t2, t3, t4) in itertools.permutations(((type2, 1), (type3, 2), (type4, 3))):
                        if t2[0] in types2 and t3[0] in types3 and t4[0] in types4:
                            if wildcard in (types1, types2, types3, types4):
                                # Workaround to be more consistent with AMBER.  It uses wildcards to define most of its
                                # impropers, which leaves the ordering ambiguous.  It then follows some bizarre rules
                                # to pick the order.
                                a1 = torsion[t2[1]]
                                a2 = torsion[t3[1]]
                                e1 = data.atoms[a1].element
                                e2 = data.atoms[a2].element
                                if e1 == e2 and a1 > a2:
                                    (a1, a2) = (a2, a1)
                                elif e1 != elem.carbon and (e2 == elem.carbon or e1.mass < e2.mass):
                                    (a1, a2) = (a2, a1)
                                force.addTorsion(a1, a2, torsion[0], torsion[t4[1]], tordef.c[0], tordef.c[1], tordef.c[2], tordef.c[3], tordef.c[4], tordef.c[5])
                            else:
                                # There are no wildcards, so the order is unambiguous.
                                force.addTorsion(torsion[0], torsion[t2[1]], torsion[t3[1]], torsion[t4[1]], tordef.c[0], tordef.c[1], tordef.c[2], tordef.c[3], tordef.c[4], tordef.c[5])
                            done = True
                            break

parsers["RBTorsionForce"] = RBTorsionGenerator.parseElement




## @private
class NonbondedGenerator:
    """A NonbondedGenerator constructs a NonbondedForce."""

    def __init__(self, forcefield, coulomb14scale, lj14scale):
        self.ff = forcefield
        self.coulomb14scale = coulomb14scale
        self.lj14scale = lj14scale
        self.typeMap = {}

    def registerAtom(self, parameters):
        types = self.ff._findAtomTypes(parameters, 1)
        if None not in types:
            values = (_convertParameterToNumber(parameters['charge']), _convertParameterToNumber(parameters['sigma']), _convertParameterToNumber(parameters['epsilon']))
            for t in types[0]:
                self.typeMap[t] = values

    @staticmethod
    def parseElement(element, ff):
        existing = [f for f in ff._forces if isinstance(f, NonbondedGenerator)]
        if len(existing) == 0:
            generator = NonbondedGenerator(ff, float(element.attrib['coulomb14scale']), float(element.attrib['lj14scale']))
            ff.registerGenerator(generator)
        else:
            # Multiple <NonbondedForce> tags were found, probably in different files.  Simply add more types to the existing one.
            generator = existing[0]
            if generator.coulomb14scale != float(element.attrib['coulomb14scale']) or generator.lj14scale != float(element.attrib['lj14scale']):
                raise ValueError('Found multiple NonbondedForce tags with different 1-4 scales')
        for atom in element.findall('Atom'):
            generator.registerAtom(atom.attrib)

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        methodMap = {NoCutoff:mm.NonbondedForce.NoCutoff,
                     CutoffNonPeriodic:mm.NonbondedForce.CutoffNonPeriodic,
                     CutoffPeriodic:mm.NonbondedForce.CutoffPeriodic,
                     Ewald:mm.NonbondedForce.Ewald,
                     PME:mm.NonbondedForce.PME}
        if nonbondedMethod not in methodMap:
            raise ValueError('Illegal nonbonded method for NonbondedForce')
        force = mm.NonbondedForce()
        for atom in data.atoms:
            t = data.atomType[atom]
            if t in self.typeMap:
                values = self.typeMap[t]
                force.addParticle(values[0], values[1], values[2])
            else:
                raise ValueError('No nonbonded parameters defined for atom type '+t)
        force.setNonbondedMethod(methodMap[nonbondedMethod])
        force.setCutoffDistance(nonbondedCutoff)
        if 'ewaldErrorTolerance' in args:
            force.setEwaldErrorTolerance(args['ewaldErrorTolerance'])
        if 'useDispersionCorrection' in args:
            force.setUseDispersionCorrection(bool(args['useDispersionCorrection']))
        sys.addForce(force)

    def postprocessSystem(self, sys, data, args):
        # Create exceptions based on bonds.
        
        bondIndices = []
        for bond in data.bonds:
            bondIndices.append((bond.atom1, bond.atom2))

        # If a virtual site does *not* share exclusions with another atom, add a bond between it and its first parent atom.

        for i in range(sys.getNumParticles()):
            if sys.isVirtualSite(i):
                (site, atoms, excludeWith) = data.virtualSites[data.atoms[i]]
                if excludeWith is None:
                    bondIndices.append((i, site.getParticle(0)))
        
        # Certain particles, such as lone pairs and Drude particles, share exclusions with a parent atom.
        # If the parent atom does not interact with an atom, the child particle does not either.
        
        for atom1, atom2 in bondIndices:
            for child1 in data.excludeAtomWith[atom1]:
                bondIndices.append((child1, atom2))
                for child2 in data.excludeAtomWith[atom2]:
                    bondIndices.append((child1, child2))
            for child2 in data.excludeAtomWith[atom2]:
                bondIndices.append((atom1, child2))

        # Create the exceptions.

        nonbonded = [f for f in sys.getForces() if isinstance(f, mm.NonbondedForce)][0]
        nonbonded.createExceptionsFromBonds(bondIndices, self.coulomb14scale, self.lj14scale)

parsers["NonbondedForce"] = NonbondedGenerator.parseElement

