import numpy as np

def passe_arriere(delta_h, a, h, W, activation):
    """ Réalise une passe arrière dans le réseau de neurones (rétropropagation)
    
    Parametres
    ----------
    delta_h : matrice contenant les valeurs du gradient du coût par rapport à la sortie du réseau
    a : liste contenant les potentiels d'entrée des couches du réseau
    h : liste contenant les sorties des couches du réseau
    W : liste contenant les matrices des poids du réseau
    activation : liste contenant les fonctions d'activation des couches du réseau

    Return
    -------
    delta_W : liste contenant les matrice des gradients des poids des couches du réseau
    delta_b : liste contenant les matrice des gradients des biais des couches du réseau

    """

    delta_b = []
    delta_W = []

    for i in range(len(W)-1,-1,-1):

        #######################
        ##### A compléter ##### 
        #######################

        delta_i = delta_h * activation[i](a[i], deriv=True)
        delta_b.append(np.expand_dims(delta_i.sum(axis=-1), axis=-1)) 
        delta_W.append(delta_i @ h[i].T)
        delta_h = W[i].T @ delta_i

    delta_W.reverse()
    delta_b.reverse()
    return delta_W, delta_b