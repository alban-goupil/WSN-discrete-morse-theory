# Théorie de Morse Discrète pour les réseaux de capteurs

## Objectifs

En plaquant un champ de vecteurs discrets sur un réseau de
capteurs, il est possible, via la théorie de Morse discrète
de Forman de récupérer la topologie de la couverture du
réseau et surtout de le suivre simplement lors de ses
modifications. Il devient alors possible de limiter le
nombre de capteurs en éveil pour garantir la couverture et
surtout d'en réveiller certain en cas de modification de la
topologie.

## Les programmes

Le répertoire [src/](./src/) contient les programmes
principaux avec une documentation dans les fichiers python.

Le répertoire [topology/](./topology/) contient une petite
librairie python pour les calculs de topologies algébriques.

Le fichier [output/output.avi](./output/output.avi) contient
une petite vidéo des résultats des algorithmes pour garantir
la couverture quand le réseau subit des pertes de capteurs.
