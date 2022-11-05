import numpy as np

def calcule_cout_mse(y, d):
    """ Calcule la valeur de la fonction cout MSE (moyenne des différences au carré)
    
    Parametres
    ----------
    y : matrice des données prédites 
    d : matrice des données réelles 
    
    Return
    -------
    cout : nombre correspondant à la valeur de la fonction cout (moyenne des différences au carré)

    """

    #######################
    ##### A compléter ##### 
    #######################

    cout = 0.5 * np.square(y - d).sum()

    return cout
