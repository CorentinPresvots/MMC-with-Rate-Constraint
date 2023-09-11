# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 19:04:56 2023

@author: presvotscor
"""

# Ouvrir le fichier en mode lecture
with open('data_test.txt', 'r') as file:
    # Lire le contenu du fichier
    content = file.read()

# Extraire la liste à partir du contenu du fichier
# En supposant que la liste est toujours dans la forme indiquée
start = content.find('[')
end = content.find(']')
if start != -1 and end != -1:
    # Extraire la partie entre crochets et la diviser en éléments individuels
    list_str = content[start + 1:end]
    x_test_RTE = [float(item.strip()) for item in list_str.split(',')]

    # Maintenant, 'elements' contient la liste que vous souhaitez récupérer
    print(x_test_RTE[0:10])
