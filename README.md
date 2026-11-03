# MATH60600-data-mining
Partie A (15 points)
Le problème de prédiction multi-classes est intéressant car il y a plusieurs façons de l’aborder. Plusieurs 
méthodes sont capables de traiter directement une variable cible ayant plus de 2 catégories. Mais il est aussi 
possible de scinder le problème en plusieurs problèmes de classification binaires. L’article suivant discute 
de ce type d’approches.
Galar, M., Fernández, A., Barrenechea, E., Bustince, H., & Herrera, F. (2011). An overview of ensemble 
methods for binary classifiers in multi-class problems: Experimental study on one-vs-one and one-vs-all 
schemes. Pattern Recognition, 44(8), 1761-1776.
Vous devez lire cet article mais vous n’êtes pas obligé d’utiliser ce type d’approches pour vos
prédictions finales. Mais vous devez essayer au moins deux méthodes (et le rapporter dans votre 
rapport) et au moins l’une d’elle doit être de ce type. Par exemple, vous pouvez essayer une forêt 
aléatoire directement sur le Y et essayer une approche one-vs-all (voir l’article) avec du boosting.
Ce travail se place dans le même contexte que l’exemple avec les données de jeux en ligne du cours. Les 
variables explicatives sont les mêmes. Voir les notes de cours pour les détails. Seulement la variable cible 
Y change. Nous supposons qu’elle mesure quel jeu le joueur a installé en premier durant le mois suivant. 
Plus précisément, Y prend les 4 valeurs suivantes :
Y=1 si aucun jeu n’a été installé
Y=2 si le jeu A a été installé
Y=3 si le jeu B a été installé
Y=4 si le jeu C a été installé
2
Pour ce travail vous devez développer un modèle de prédiction pour ce Y. Pour ce faire, vous avez un 
échantillon d’entrainement de taille 5000. Vous devez fournir des prédictions pour 55000 nouvelles 
observations. Le taux de bonne classification (accuracy) est le critère de performance.
Notez que comme vous allez vous familiariser avec l’optimisation des hyperparamètres dans la partie B de 
ce travail, il pourrait être pertinent d’utiliser, au moins minimalement, ce que vous allez apprendre. 
Vous devez expliquer clairement et de manière concise les étapes de votre démarche (méthodologie) dans 
votre rapport et décrire le modèle que vous avez retenu. Le code doit être annoté et compréhensible. Bien 
indiquer au début du code quels sont les packages requis ou les librairies requises. Vous pouvez utiliser soit 
R, soit Python ou les deux.
Critères pour la correction de la partie A:
1) Méthodologie (5 points)
2) Présentation générale du rapport et du code (5 points)
3) Performance du modèle (5 points)
Nous allons utiliser la plate-forme Kaggle. Le lien pour accéder à la compétition est :
https://www.kaggle.com/competitions/travail-equipe-automne-2026
Au moins un des membres de l’équipe doit avoir un compte Kaggle. Notez bien que la performance dans le 
tableau Public, lorsque vous soumettez des prédictions, est calculée avec 9% des 55000 observations tests. 
Votre performance finale sera évaluée à la fin de la compétition avec l’autre 91% et apparaitra une semaine
après la date de remise sur le tableau Private. Ceci va vous éviter de faire du sur-ajustement sur les données 
tests. 
Voici un petit exemple simple où j’entraine une forêt aléatoire de base (paramètres par défaut) avec le 
package ranger. J’obtiens ensuite les prédictions sur les données test et je prépare le fichier .csv pour la 
soumission sur Kaggle.
Partie B (15 points)
Cette partie est au sujet de l’optimisation des hyperparamètres d’un modèle de prédiction. Le travail consiste 
à :
1/ Trouver au moins 3 ressources (articles, livres, sites web, etc.) pertinentes et complémentaires qui 
fournissent une introduction à ce sujet. Votre rapport doit présenter ces ressources, expliquer ce qu’elles 
contiennent et pourquoi elles sont pertinentes et complémentaires.
2/ Décrire 3 grandes méthodes pour faire de l’optimisation des hyperparamètres.
3/ Appliquer l’une d’elle avec les données de l’exemple de jeux en ligne du cours (avec le Y continu) en 
utilisant un package R ou une librairie Python de votre choix. Le code doit être annoté et compréhensible. 
Bien indiquer au début du code quels sont les packages requis ou les librairies requises
Critères pour la correction de la partie B:
1) Qualité, pertinence et complémentarité des ressources identifiées (5 points)
2) Description des 3 méthodes (5 points)
3) Qualité de l’exemple avec les données du cours (5 points)
