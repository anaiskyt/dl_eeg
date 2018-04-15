Description des fichiers :

- cnn_autoencoder, lstm_autoencoder : Autoencoders entrainés
- net_with_encoder : réseau constitué de l'encoder + couches convolutionnelles
- test_net : test de l'encoder avec algorithme KMeans
- visualize_data : visualisation des données MEG basée sur la librairie mne (référence pour le traitement des données MEG en Python)

Utils :
- extract_matricies + data_convertion : Extraction des données du fichier .mat pour les transformer en matrices exploitables
- data_augmentation : augmentation des données par sous-échantillonnage

Structure globale de chaque fichier de données :
- label
Contient tous les noms de capteurs utilisés.
Taille 241
- trial
Contient toutes les données des mesures.
Liste des trials
Dans chaque liste, matrice (nb de capteurs * indice temporel)
- time
Chaque pas de temps auquel une mesure a été effectuée
- trialinfo
Liste de taille le nombre de trials.
Contient l'information de la séquence captée pour chaque trial.
- fsample
Fréquence d'échantillonnage
- cfg
 version
 trackcallinfo
 trackconfig
 checkconfig
 checksize
 showcallinfo
 debug
 trackdatainfo
 trackparaminfo
 previous
- grad
balance
Supine
 labelorg
    Liste des capteurs
 labelnew
 tra
    (271, 271)
invcomp
 tra
 labelorg
 labelnew
 chantypeorg
 chantypenew
 chanunitorg
 chanunitnew
pca
 tra
 labelorg
 labelnew
 chantypeorg
 chantypenew
 chanunitorg
 chanunitnew
current
previous
chanori
chanpos
chantype
chanunit
coilori
coilpos
label
labelorg
tra
type
unit
