import numpy as np

def lecture_donnees(nom_fichier, delimiteur=','):
    """ Lit le fichier contenant les données et renvoiee les matrices correspondant

    Parametres
    ----------
    nom_fichier : nom du fichier contenant les données
    delimiteur : caratère délimitant les colonne dans le fichier ("," par défaut)

    Retour
    -------
    x : matrice des données de dimension [N, nb_var]
    d : matrice contenant les valeurs de la variable cible de dimension [N, nb_cible]
    N : nombre d'éléments
    nb_var : nombre de variables prédictives
    nb_cible : nombre de variables cibles

    """
    
    data = np.loadtxt(nom_fichier, delimiter=delimiteur)

    nb_cible = 1
    nb_var = data.shape[1] - nb_cible
    N = data.shape[0]

    x = data[:, :-1]
    d = data[:, nb_var:].reshape(N,1)
    
    return x, d, N, nb_var, nb_cible