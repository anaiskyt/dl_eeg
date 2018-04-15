# dl_eeg

Structure du TFLA.mat
Key: __header__
Key: __version__
Key: __globals__
Key: data
	Field: label
	Field: trial
	Field: time
	Field: trialinfo
	Field: fsample
	Field: cfg
	Field: grad


Labels : 241
4 labels finaux pour les EMG sur les muscles


len(processed_data['data']['trial'][0][0][0]) = 384



1.0 70
5.0 71
4.0 73
2.0 71
6.0 94

(289, 1221, 241)
7

 data
	 label
        Contient tous les noms de capteurs utilisés.
        Taille 241
	 trial
	    Contient toutes les données des mesures.
	    Liste des trials
	    Dans chaque liste, matrice (nb de capteurs * indice temporel)
	 time
	    Chaque pas de temps auquel une mesure a été effectuée
	 trialinfo
	     Liste de taille le nombre de trials.
	     Contient l'information de la séquence captée pour chaque trial.
	 fsample
	 cfg
		 version
			 name
			 id
		 trackcallinfo
		 trackconfig
		 checkconfig
		 checksize
		 showcallinfo
		 debug
		 trackdatainfo
		 trackparaminfo
		 previous
	 grad
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