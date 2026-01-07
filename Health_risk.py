# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %%
#prepare data

read_csv = pd.read_csv('https://raw.githubusercontent.com/ChickenMan-1080/Health-risk/refs/heads/main/Lifestyle_and_Health_Risk_Prediction_Synthetic_Dataset.csv')
df = pd.DataFrame(read_csv)
#print(df)


# %%            
#Data exploration

print(df.head())
print('--'*50)
#%%
print(df.info())
print('--'*50)
#%%
print(df.describe())
print('--'*50)
print(df.keys())


# %%
#Explore y variable


print(df['health_risk'])

plt.figure(figsize = (8,6))
sns.countplot(data = df , x = 'health_risk')
plt.title('Distribution of Health Risk')
plt.xlabel('Risk')
plt.ylabel('Distribution')
plt.grid(True)

plt.show()

#%%
#Data selection
#Data analysis (multivariate analysis) and outlier check

#age and health risk

plt.figure(figsize = (8,6))
sns.boxplot(data=df, x="age", y="health_risk")
plt.title('Distribution of age')
plt.xlabel('age')
plt.ylabel('Health Risk')
plt.grid(True)

plt.show()

#%% 
#bmi and health risk

plt.figure(figsize = (8,6))
sns.boxplot(data=df, x="bmi", y="health_risk")
plt.title('Distribution of bmi')
plt.xlabel('bmi')
plt.ylabel('Health Risk')
plt.grid(True)

plt.show()

#%%

#sleep and health risk

plt.figure(figsize = (8,6))
sns.boxplot(data=df, x="sleep", y="health_risk")
plt.title('Distribution of sleep')
plt.xlabel('sleep')
plt.ylabel('Distribution')
plt.grid(True)

plt.show()

#%%
#exercise and health risk

plt.figure(figsize = (8,6))
sns.countplot(data = df , x = 'exercise', hue = 'health_risk' , order = ['none', 'low', 'medium', 'high'])
plt.title('Distribution of exercise')
plt.xlabel('exercise')
plt.ylabel('count')
plt.grid(True)

plt.show()

#%%
#sugar_intake and health risk

plt.figure(figsize = (8,6))
sns.countplot(data = df , x = 'sugar_intake', hue = 'health_risk' , order = ['low', 'medium', 'high'])
plt.title('Distribution of sugar_intake')
plt.xlabel('sugar_intake')
plt.ylabel('count')
plt.grid(True)

plt.show()

#%%
#smoking and health risk

plt.figure(figsize = (8,6))
sns.countplot(data = df , x = 'smoking', hue = 'health_risk' , order = ['no', 'yes'])
plt.title('Distribution of smoking')
plt.xlabel('smoking')
plt.ylabel('count')
plt.grid(True)

plt.show()

#%%
#alcohol and health risk

plt.figure(figsize = (8,6))
sns.countplot(data = df , x = 'profession', hue = 'health_risk')
plt.title('Distribution of profession')
plt.xlabel('profession')
plt.ylabel('count')
plt.grid(True)

plt.show()

#%%
#profession and health risk

plt.figure(figsize = (8,6))
sns.countplot(data = df , x = 'profession', hue = 'health_risk')
plt.title('Distribution of profession')
plt.xlabel('profession')
plt.ylabel('count')
plt.grid(True)

plt.show()

#%%
#married and health risk

plt.figure(figsize = (8,6))
sns.countplot(data = df , x = 'married', hue = 'health_risk')
plt.title('Distribution of married')
plt.xlabel('married')
plt.ylabel('count')
plt.grid(True)

plt.show()
#%%
#height and health risk

plt.figure(figsize = (8,6))
sns.boxplot(data = df , x = 'height', y = 'health_risk')
plt.title('Distribution of height')
plt.xlabel('height')
plt.ylabel('health risk')
plt.grid(True)

plt.show()
#%%
#weight and health risk

plt.figure(figsize = (8,6))
sns.boxplot(data = df , x = 'weight', y = 'health_risk')
plt.title('Distribution of weight')
plt.xlabel('weight')
plt.ylabel('health risk')
plt.grid(True)

plt.show()



#%%
#Label encoding
#setup data

df_final = df

risk_map = {'low':0,'high':1}
df_final['health_risk'] = df_final['health_risk'].map(risk_map)


exercise_map = {'none':0,'low':1,'medium':2,'high':3}
df_final['exercise'] = df_final['exercise'].map(exercise_map)


sugar_intake_map = {'low':0,'medium':1,'high':2}
df_final['sugar_intake'] = df_final['sugar_intake'].map(sugar_intake_map)


smoking_map = {'no':0,'yes':1}
df_final['smoking'] = df_final['smoking'].map(smoking_map)


alcohol_map = {'no':0,'yes':1}
df_final['alcohol'] = df_final['alcohol'].map(alcohol_map)


profession_map = {'office_worker':0,'teacher':1,'artist':2,'farmer':3,'driver':4,'engineer':5,'student':6,'doctor':7}
df_final['profession'] = df_final['profession'].map(profession_map)

married_map = {'no':0,'yes':1}
df_final['married'] = df_final['married'].map(married_map)


print(df_final)


#%%
#training setup 

df_model = df_final.dropna()
X = df_model.drop(columns=['health_risk'])
y = df_model['health_risk']

#%%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

#%%
#Test 

y_pred = rf.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")
print("\n--- Classification Repor   t ---")
print(classification_report(y_test, y_pred))

#%%
#VIF

X_vif = df_final.drop(columns=['health_risk']).dropna()
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
print(vif_data.sort_values(by="VIF", ascending=False))

#%%
#create new data frame and delete high VIF

df_final2 = df_final.copy()
df_final2.drop(columns=['weight', 'height'], inplace=True)

#%%

print(df_final2.columns)

#%%
df_model2 = df_final2.dropna()
X_2 = df_model2.drop(columns=['health_risk'])
y_2 = df_model2['health_risk']

#%%
X_train, X_test, y_train, y_test = train_test_split(X_2, y_2, test_size=0.3, stratify=y_2, random_state=42)


rf_final = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_final.fit(X_train, y_train)


y_pred = rf_final.predict(X_test)
print(classification_report(y_test, y_pred))

#%%
#VIF

X_vif = df_final2.drop(columns=['health_risk']).dropna()
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
print(vif_data.sort_values(by="VIF", ascending=False))




# %%
#create new data frame anddelete high VIF(final)

df_final3 = df_final2.copy()
df_final3.drop(columns=['sleep'], inplace=True)

#%%

print(df_final3.columns)

# %%
df_model3 = df_final3.dropna()
X_3 = df_model3.drop(columns=['health_risk'])
y_3 = df_model3['health_risk']

#%%
X_train, X_test, y_train, y_test = train_test_split(X_3, y_3, test_size=0.3, stratify=y_3, random_state=42)


rf_final = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_final.fit(X_train, y_train)


y_pred = rf_final.predict(X_test)
print(classification_report(y_test, y_pred))

#%%
#VIF

X_vif = df_final3.drop(columns=['health_risk']).dropna()
vif_data = pd.DataFrame()
vif_data["Feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
print(vif_data.sort_values(by="VIF", ascending=False))


# %%
#save model

joblib.dump(rf_final, 'health_risk_model.pkl')

joblib.dump(X_3.columns.tolist(), 'model_columns.pkl')


