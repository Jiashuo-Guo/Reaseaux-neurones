import numpy as np

def decoupage_donnees(x,d,prop_val=0.2, prop_test=0.2):
    """ Découpe les données initiales en trois sous-ensembles distincts d'apprentissage, de validation et de test
    
    Parametres
    ----------
    x : matrice des données de dimension [N, nb_var]
    d : matrice des valeurs cibles [N, nb_cible]
    prop_val : proportion des données de validation sur l'ensemble des données (entre 0 et 1)
    prop_test : proportion des données de test sur l'ensemble des données (entre 0 et 1)
    
    avec N : nombre d'éléments, nb_var : nombre de variables prédictives, nb_cible : nombre de variables cibles

    Retour
    -------
    x_app : matrice des données d'apprentissage
    d_app : matrice des valeurs cibles d'apprentissage
    x_val : matrice des données d'apprentissage
    d_val : matrice des valeurs cibles d'apprentissage
    x_test : matrice des données d'apprentissage
    d_test : matrice des valeurs cibles d'apprentissage

    """
    #######################
    ##### A compléter ##### 
    #######################

    N = x.shape[0]
    random_index = np.arange(N)
    np.random.shuffle(random_index)
    x = x[random_index, :]
    d = d[random_index, :]

    l1 = int(N * (1 - prop_val - prop_test))
    l2 = int(N * (1 - prop_test))
    x_app = x[:l1, :]
    d_app = d[:l1, :]
    x_val = x[l1:l2, :]
    d_val = d[l1:l2, :]
    x_test = x[l2:, :]
    d_test = d[l2:, :]

    return x_app, d_app, x_val, d_val, x_test, d_test
    

