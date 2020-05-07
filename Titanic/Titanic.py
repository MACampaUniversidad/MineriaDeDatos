# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

import pandas 
import os
import matplotlib.pyplot as mathPlot
import numpy 
import seaborn as sea
import difflib 
#define a class to help us with the process
class ModelFileHelper(object):
    """Ayuda a dar una descripcion de un fichero y a su carga """
    def __init__(self, csvFile):
     self.csvFile= pandas.read_csv(csvFile)
     self.fileName=csvFile
    def getDescription(self):
        return self.csvFile.describe()

    def getModelTypeDetail(self):
        """Retorna una estructura legible con los tipos de dato del conjunto de datos del csv cargado"""
        return self.__translateTypestoHumanReadable(self.csvFile.dtypes)

    def findDifferences(self, other):
        """Retorna una lista con la comparacion de las columnas y los tipos de dos csv"""
        returnlist = list (difflib.Differ().compare(self.getModelTypeDetail().to_string().splitlines(1), other.getModelTypeDetail().to_string().splitlines(1)))
        returnlist.append("Comparativa de tamaños: ")
        returnlist.append (self.fileName +  " Filas:" + ''.join(self.__tuplaCleanUp(self.csvFile.shape[0:1])) + " Columnas:" +  ''.join(self.__tuplaCleanUp(self.csvFile.shape[1:2])))
        returnlist.append (other.fileName + " Filas:" + ''.join(self.__tuplaCleanUp(other.csvFile.shape[0:1])) + " Columnas:" +  ''.join(self.__tuplaCleanUp(other.csvFile.shape[1:2])))
        return returnlist

    def __tuplaCleanUp(self, tupla):
        result = str(tupla).replace('(','').replace(')','').replace(',','')
        return result

    def __translateTypestoHumanReadable(self, text):
        return text.replace("int64", "Numero").replace("object", "Cadena de texto AlfaNumerica").replace("float64", "Numero (largo)")

#load the train model and store it in a dictionary :
files = { "train" : ModelFileHelper("dataInputs/train.csv"), "test" :   ModelFileHelper("dataInputs/test.csv")} 
 
#describe both files 
print ("*****************************************************************")
print ("Análisis comparativo de tipos")
print ("*****************************************************************")
for index, (clave, valor) in enumerate (files.items()):
    print ("...............................................................")
    print ("File: " + valor.fileName)
    print (valor.getModelTypeDetail())
print ("----------------------------------------------------------------")
print ("Buscando diferencias entre tipos: - Significa eliminado, + significa añadido:")   
for listItem in enumerate (files.get("train").findDifferences(files.get("test"))):
    print (listItem)
#check for null fields:
print ("*****************************************************************")
print ("Buscando campos vacios:")   
print ("*****************************************************************")
for index, (clave, valor) in enumerate (files.items()):
    print ("...............................................................")
    print ("File: " + valor.fileName)
    valor.csvFile.info()
    print (valor.getDescription())  

# %%
#Correlations
print ("*****************************************************************")
print ("Limpieza del modelo: análisis de las correlaciones ")   
print ("*****************************************************************")
files.get("train").csvFile.corr()



# %%
#Clean up Cabin Column
print ("*****************************************************************")
print ("Limpieza del modelo: Eliminando campos Irrelevantes ")   
print ("*****************************************************************")
for index, (clave, valor) in enumerate (files.items()):
    print ("...............................................................")
    print ("File: " + valor.fileName)
    print ("Removing Column Cabin: ")
    valor.csvFile.drop('Cabin', axis=1, inplace=True)
    print ("Removing Column Embarked: ")
    valor.csvFile.drop('Embarked', axis=1, inplace=True)
    print (valor.getModelTypeDetail())  


# %%
#Infer missing age Data
print ("*****************************************************************")
print ("Limpieza del modelo: Inferir campos de Edad vacios:")   
print ("*****************************************************************") 
files.get("train").csvFile.query('Age >=1').groupby(['Survived','Pclass', 'Sex']).agg({'Age': ['mean', 'min', 'max']})


# %%
#Comparative of age groups between train model and test model
print ("*****************************************************************")
print ("Limpieza del modelo: comparacion de grupos de edad :")   
print ("*****************************************************************") 
print ("modelo Train:")
files.get("train").csvFile.query('Age >=1').groupby(['Pclass', 'Sex']).agg({'Age': ['mean', 'min', 'max']})


# %%
#imp
print ("modelo test:")
files.get("test").csvFile.query('Age >=1').groupby(['Pclass', 'Sex']).agg({'Age': ['mean', 'min', 'max']})


# %%
fig, axs = mathPlot.subplots(ncols=2, figsize=(30,5))
sea.pointplot(x="Pclass", y="Age", hue="Sex", data=  files.get("train").csvFile.query('Age >= 1'), ax=axs[0])
sea.pointplot(x="Pclass", y="Age", hue="Sex", data=  files.get("test").csvFile.query('Age >= 1'), ax=axs[1])


# %%
print ("*****************************************************************")
print ("Limpieza del modelo: corrección de de la edad :")   
print ("*****************************************************************") 
print ("modelo Train:")
#estadisticasTrain= files.get("train").csvFile.query('Age >=1').groupby(['Survived','Pclass', 'Sex']).agg({'Age': ['mean', 'min', 'max']})
nan= numpy.nan
print (files.get("train").csvFile.query('(Age.isnull()) | (Age < 0)', engine='python'))


# %%


