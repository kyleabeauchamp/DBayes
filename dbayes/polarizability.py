import re
import numpy as np
import pandas as pd
import mdtraj as md
import simtk.unit as u
import simtk.openmm.app.element as element

baseline = 0.32
elemental_coefficients = dict(C=1.51, H=0.17, O=0.57, N=1.05, S=2.99, P=2.48, F=0.22, Cl=2.16, Br=3.29, I=5.45)

def polarizability(traj, add_baseline=True):
    """Estimate the polarizabilty of a simulation box using the simple
    elemental regression model (e.g. counting elements) of Sales, 2002.
    Units will be in cubic nanometers.
    """

    elements = [a.element.symbol for a in traj.top.atoms]

    alpha = sum(elemental_coefficients[e] for e in elements)

    if add_baseline:
        alpha += baseline

    alpha *= u.angstrom ** 3

    return alpha


def dielectric_correction(traj):
    """Estimate the polarizabilty of a simulation box using the simple
    """
    return 4 * np.pi * polarizability(traj) / traj.unitcell_volumes.mean()

def polarizability_from_formula(formula, add_baseline=True):
    element_dict = formula_to_element_counts(formula)
    
    alpha = sum(number * elemental_coefficients[e] for e, number in element_dict.items())

    if add_baseline:
        alpha += baseline
        
    alpha *= u.angstrom ** 3
    
    return alpha

def dielectric_correction_from_formula(formula, mass_density, add_baseline=True):
    alpha = polarizability_from_formula(formula, add_baseline)

    element_dict = formula_to_element_counts(formula)

    molar_mass = np.sum([number * element.Element.getBySymbol(e).mass for e, number in element_dict.items()])
    molar_mass *= (u.kilograms / u.dalton)
    molar_mass /= (u.AVOGADRO_CONSTANT_NA * u.mole)
    
    molar_volume = (molar_mass / mass_density)

    return 4 * np.pi * alpha / molar_volume
    

def formula_to_element_counts(test):
    pattern = r'([A-Z][a-z]{0,2}\d*)'
    pieces = re.split(pattern, test)
    print "\ntest=%r pieces=%r" % (test, pieces)
    data = pieces[1::2]
    rubbish = filter(None, pieces[0::2])
    pattern2 = r'([A-Z][a-z]{0,2})'

    results = {}
    for piece in data:
        print(piece)
        element, number = re.split(pattern2, piece)[1:]
        try:
            number = int(number)
        except ValueError:
            number = 1
        results[element] = number

    return results


element.Element.getBySymbol("Cl").mass
