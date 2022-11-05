import numpy as np


def calcule_cout_cee(y, d):
    """ Calcule la valeur de la fonction cout CEE (entropie croisée)

    Parametres
    ----------
    y : matrice des données prédites
    d : matrice des données réelles

    Return
    -------
    cout : nombre correspondant à la valeur de la fonction cout (entropie croisée)

    """

    #######################
    ##### A compléter #####
    #######################

    cout = - (d * np.log(y + 1e-7)).sum()

    return cout