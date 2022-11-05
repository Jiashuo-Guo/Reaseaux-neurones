import numpy as np


def softmax(z, deriv=False):
    """ Calcule la valeur de la fonction softmax ou de sa dérivée appliquée à z

    Parametres
    ----------
    z : peut être un scalaire ou une matrice
    deriv : booléen. Si False renvoie la valeur de la fonction sigmoide, si True renvoie sa dérivée

    Return
    -------
    s : valeur de la fonction sigmoide appliquée à z ou de sa dérivée. Même dimension que z

    """

    s = np.exp(z) / np.exp(z).sum(axis=0, keepdims=True)
    if deriv:
        return np.array(1)
    else:
        return s

