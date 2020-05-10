# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas 
import numpy as np 
import os
import matplotlib.pyplot as mathPlot
import seaborn as sea
import difflib 
from sklearn.model_selection import train_test_split  
#define a class to help us with the process
class ModelFileHelper(object):
    """Ayuda a dar una descripcion de un fichero y a su carga """
    def __init__(self, csvFile):
     self.csvFile= pandas.read_csv(csvFile) 
     self.fileName=csvFile
    def getDescription(self):
        return self.csvFile.describe()

    def dropColumn(self, columnName):
        """wrapper para eliminar una columna"""
        self.csvFile= self.csvFile.drop(columnName, axis=1)

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

    def pearson(self,  A,  B):
        """ indice de coorrelacion lineal de Pearson de la variable A con respecto a variable B. 
        Acotado entre [1 , -1] indicando |1| alta coorrelacion y en el caso de ser negativo el coeficiente, correlación inversa """
        pearson = self.csvFile[A].corr(self.csvFile[B])
        direccion = "directa" if (pearson>0) else ("inversa" if pearson < 0 else "No existe")
        return "Correlación lineal [" + direccion + "]: " + str( abs(pearson) )

    def exportHarmonizatedModel(self, harmonizationMatrix, harmonizationquery, fileName):
        """Exporta el modelo tras armonizar los valores en funcion de una matriz de armonización dada y una query"""
        harmonizated =self.csvFile 
        dataframe = pandas.DataFrame(harmonizationMatrix)
        for index, trainedRow in dataframe.iterrows() :
            group =harmonizated.query(harmonizationquery) 
            for index, groupRow in  group.iterrows():
                randomVal= np.random.randint(trainedRow['Min'], trainedRow['Max'])
                harmonizated.loc[(harmonizated.PassengerId  ==  groupRow.PassengerId) , "Age"]=randomVal
        #dump to csv
        print ("volcando a archivo harmonizated_train.csv")
        harmonizated.to_csv(fileName,  index=False)  


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
#Clean up Cabin & embarked Columns
print ("*****************************************************************")
print ("Limpieza del modelo: Eliminando campos Irrelevantes ")   
print ("*****************************************************************")
for index, (clave, valor) in enumerate (files.items()):
    print ("...............................................................")
    print ("File: " + valor.fileName)
    print ("Removing Column Cabin: ")
    valor.dropColumn('Cabin')
    print ("Removing Column Embarked: ")
    valor.dropColumn('Embarked')
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

print ("modelo test:")
files.get("test").csvFile.query('Age >=1').groupby(['Pclass', 'Sex']).agg({'Age': ['mean', 'min', 'max']})


# %%
fig, axs = mathPlot.subplots(ncols=2, figsize=(30,5))

sea.pointplot(x="Pclass", y="Age", hue="Sex", data=  files.get("train").csvFile.query('Age >= 1'), ax=axs[0])
sea.pointplot(x="Pclass", y="Age", hue="Sex", data=  files.get("test").csvFile.query('Age >= 1'), ax=axs[1])


# %%
print ("*****************************************************************")
print ("Limpieza del modelo: corrección de la edad :")   
print ("*****************************************************************") 
print ("modelo Train:")
estadisticasTrain= files.get("train").csvFile.query('Age >=1').groupby(['Survived','Pclass', 'Sex']).agg({'Age': ['mean', 'min', 'max']})
#ya vamos a trabajar con un modelo exportable usando las estadisticas obtenidas con anterioridad 
#Survived	Pclass	Sex			mean	min		max
#0			1		female	25.666667	2.0		50.0
#0			1		male	44.581967	18.0	71.0
#0			2		female	36.000000	24.0	57.0
#0			2		male	33.369048	16.0	70.0
#0			3		female	23.818182	2.0		48.0
#0			3		male	27.255814	1.0		74.0
#1			1		female	34.939024	14.0	63.0
#1			1		male	37.153846	4.0		80.0
#1			2		female	28.080882	2.0		55.0
#1			2		male	19.833333	1.0		62.0
#1			3		female	20.155556	1.0		63.0
#1			3		male	22.864865	1.0		45.0

estadisticasTrain = [{'Survived':0,'Pclass':1,'Sex':'female','Min':2,'Max':50},
{'Survived':0,'Pclass':1,'Sex':'male','Min':18,'Max':71},
{'Survived':0,'Pclass':2,'Sex':'female','Min':24,'Max':57},
{'Survived':0,'Pclass':2,'Sex':'male','Min':16,'Max':70},
{'Survived':0,'Pclass':3,'Sex':'female','Min':2,'Max':48},
{'Survived':0,'Pclass':3,'Sex':'male','Min':1,'Max':74},
{'Survived':1,'Pclass':1,'Sex':'female','Min':14,'Max':63},
{'Survived':1,'Pclass':1,'Sex':'male','Min':4,'Max':80},
{'Survived':1,'Pclass':2,'Sex':'female','Min':2,'Max':55},
{'Survived':1,'Pclass':2,'Sex':'male','Min':1,'Max':62},
{'Survived':1,'Pclass':3,'Sex':'female','Min':1,'Max':63},
{'Survived':1,'Pclass':3,'Sex':'male','Min':1,'Max':45}]
#Usamos las funciones de exportación que realizan van a completar los valores faltantes con valores aleatorios entre
#el máximo y el minimo de la matriz de armonizacion. 
files.get("train").exportHarmonizatedModel(estadisticasTrain,
'(@pandas.isnull(Age) or (Age < 1)) and (Survived == @trainedRow.Survived) and (Pclass == @trainedRow.Pclass) and (Sex == @trainedRow.Sex)', 
"harmonizated_train.csv" )

files.get("test").exportHarmonizatedModel(estadisticasTrain,
'(@pandas.isnull(Age) or (Age < 1)) and (Pclass == @trainedRow.Pclass) and (Sex == @trainedRow.Sex)', 
"harmonizated_test.csv" )


# %%
print ("*****************************************************************")
print ("Regresión lineal Edad - Supervivencia")   
print ("*****************************************************************") 
fig,axs = mathPlot.subplots(ncols=2, figsize=(30,5))
dfTrain=files.get("train").csvFile
dfHarmoTrain=files.get("harmo_train").csvFile
sea.regplot(x="Age", y="Survived", data=dfTrain   , ax=axs[0])
sea.regplot(x="Age", y="Survived", data= dfHarmoTrain   , ax=axs[1])

# %%
print ("*****************************************************************")
print ("Regresiónes lineales respecto al Sexo (Train original)")   
print ("*****************************************************************") 
sea.set(style="ticks", color_codes=True)
sea.pairplot(dfTrain,  hue="Sex", palette="husl",kind="reg")

# %%
print ("*****************************************************************")
print ("Regresiónes lineales respecto al Sexo (Train armonizado)")   
print ("*****************************************************************") 
sea.set(style="ticks", color_codes=True)
print ("modelo Armonizado")
sea.pairplot(dfHarmoTrain,  hue="Sex", palette="BrBG",kind="reg")

# %%
#refrescar los modelos (recuperar columnas eliminadas)
files["harmo_test"]= ModelFileHelper('harmonizated_test.csv')
files["harmo_train"]= ModelFileHelper('harmonizated_train.csv')
dfHarmoTest=files.get("harmo_test").csvFile
dfHarmoTrain=files.get("harmo_train").csvFile

dfHarmoTrain['Sex'].replace(["female","male"],[0,1], inplace=True) 
dfHarmoTest['Sex'].replace(["female","male"],[0,1], inplace=True) 
#volver a sacar las columnas ticket y name.
dfHarmoTrain.drop('Ticket', axis=1, inplace=True)
dfHarmoTrain.drop('Name', axis=1, inplace=True)
#el id de pasajero no nos aporta nada en el modelo, no interviene por lo que lo eliminamos.
dfHarmoTrain.drop('PassengerId', axis=1, inplace=True)

dfHarmoTest.drop('Ticket', axis=1, inplace=True)
dfHarmoTest.drop('Name', axis=1, inplace=True)

#reajustar los tipos del fare y eliminar potenicales nulos
print("reajuste de Fare para eliminar posibles nulos y transformacion a tipo entero")
 
dfHarmoTrain['Fare'] = dfHarmoTrain['Fare'].fillna(0)
dfHarmoTrain['Fare'] = dfHarmoTrain['Fare'].astype(int)
dfHarmoTest['Fare'] = dfHarmoTest['Fare'].fillna(0)
dfHarmoTest['Fare'] = dfHarmoTest['Fare'].astype(int)

#Trozeado del modelo. (Slicing)

xTrain = dfHarmoTrain.drop("Survived", axis=1)
yTrain = dfHarmoTrain["Survived"]
#aqui obtenemos un df sin el passenger Id pero no lo eliminamos ya que lo vamos a necesitar en
#la prediccion del submit 
xTest  = dfHarmoTest.drop("PassengerId", axis=1).copy() 
print ("Generados splits")
# %%
print ("******************************************************************************************************")
print ("Selección del mejor algoritmo de predicción")
print ("******************************************************************************************************") 
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def getResultado (y_pred):
    return pandas.DataFrame({"PassengerId" : dfHarmoTest["PassengerId"], 'Survived': yPred})

#Crear los clasificadores
xgb=XGBClassifier(n_estimators=500,n_jobs =16,max_depth=16)
logReg= LogisticRegression()
rForest= RandomForestClassifier(n_estimators=500,n_jobs =16,max_depth=16)
pctron = Perceptron(max_iter=2000 )
decTree = DecisionTreeClassifier(max_depth=16)
gaussNb= GaussianNB()

print ("Procesando XGBoost...")   
xgb.fit(xTrain,yTrain)
yPred = xgb.predict(xTest)
resultado= getResultado(yPred)
precisiones= [('XGBoost',  str(round (xgb.score(xTrain, yTrain)*100,2 )), resultado.copy())]
#-------
print ("Procesando Regresión logística...")   
logReg.fit(xTrain,yTrain)
yPred = logReg.predict(xTest)
resultado= getResultado(yPred)
precisiones.append(('Regresión Logística',  str(round (logReg.score(xTrain, yTrain)*100,2 )), resultado.copy()))
#-------
print ("Procesando Random Forest...")
rForest.fit(xTrain,yTrain)
yPred = rForest.predict(xTest)
resultado= getResultado(yPred)
precisiones.append(('Random Forest',  str(round (rForest.score(xTrain, yTrain)*100,2 )), resultado.copy()))
#-------
print ("Procesando Perceptron...")   
#-------
pctron.fit(xTrain,yTrain)
yPred = pctron.predict(xTest)
resultado= getResultado(yPred)
precisiones.append(('Perceptron',  str(round (pctron.score(xTrain, yTrain)*100,2 )), resultado.copy()))
#-------
print("Procesando árboles de decisión...")
decTree.fit(xTrain,yTrain)
yPred = decTree.predict(xTest)
resultado= getResultado(yPred)
precisiones.append(('árboles de decisión',  str(round (decTree.score(xTrain, yTrain)*100,2 )), resultado.copy()))
#-------
print ("Procesando Naybe Bayes...")
gaussNb.fit(xTrain,yTrain)
yPred = gaussNb.predict(xTest)
resultado= getResultado(yPred)
precisiones.append(('Naybe Bayes',  str(round (gaussNb.score(xTrain, yTrain)*100,2 )), resultado.copy()))
#presentar los resultados
print ("------------------------------------------------------")
print ("Precisiones del los modelos:" )
print ("------------------------------------------------------")
#ordenar de mayir a menor: 
precisiones.sort(key=lambda tupla: tupla[1], reverse=True) 
for pair in precisiones:
    print(pair[0] + " " + pair [1])
print("exportando el mejor de los modelos:")
tupla = precisiones[0]
print(tupla[0]+ "Es el ganador!")
tupla[2].to_csv( tupla[0] +"_submission.csv", index=False)
print ("Generado " +tupla[0] + "_submission.csv" )
