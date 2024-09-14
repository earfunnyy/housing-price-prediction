#!/usr/bin/env python
# coding: utf-8

# # Housing Price Prediction

# ## Library Import

# Melakukan import untuk library yang akan digunakan

# In[511]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# ## Dataset

# Menyiapkan dataset yang digunakan ke dalam variabel

# In[512]:


df = pd.read_csv('Housing.csv')


# In[513]:


df.sample(15)


# ## EDA

# Mencari informasi dari dataset yang digunakan

# In[514]:


df.info()


# In[515]:


df.describe()


# In[516]:


binary_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
df[binary_columns] = df[binary_columns].apply(lambda x: x.map({'yes': 1, 'no': 0}))


# In[517]:


df.isna().sum()


# In[518]:


df.duplicated().sum()


# In[543]:


plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", linewidths=0.5, cbar=True)
plt.show()


# ### Feature Count

# Menampilkan jumlah unique value dari masing-masing kolom

# In[519]:


bedrooms_count = df['bedrooms'].value_counts()
bedrooms_count


# In[520]:


count_bathrooms = df['bathrooms'].value_counts()
count_bathrooms


# In[521]:


stories_count = df['stories'].value_counts()
stories_count


# In[522]:


count_mainroad=df['mainroad'].value_counts()
count_mainroad


# In[523]:


guestroom_count = df['guestroom'].value_counts()
guestroom_count


# In[524]:


furnishingstatus_count = df['furnishingstatus'].value_counts()
furnishingstatus_count


# In[525]:


prefarea_count = df['prefarea'].value_counts()
prefarea_count


# ## Data Preprocessing

# Menyiapkan feature-feature untuk dimasukkan ke dalam model

# In[526]:


df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)


# In[527]:


df.head()


# In[528]:


X = df.drop('price', axis=1)  # Features
y = df['price']               # Target (Price)


# In[529]:


X


# In[530]:


y


# In[531]:


scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1, 1))


# In[532]:


X


# In[533]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Model Training

# Membuat model Random Forest untuk melatih dataset

# In[534]:


baseline_model = RandomForestRegressor(n_estimators=100, random_state=42)
baseline_model.fit(X_train, y_train)


# In[535]:


y_pred_rf = baseline_model.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Performance:")
print(f"MSE: {mse_rf}")
print(f"R²: {r2_rf}")


# ## Hyperparameter Tuning

# Melakukan tuning hyperparameter untuk mencari model terbaik

# In[536]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Definisi parameter untuk RandomizedSearchCV
param_distributions = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Inisiasi Random Forest Regressor
rf = RandomForestRegressor()

# Inisiasi RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_distributions,
    n_iter=50,  # Jumlah kombinasi yang akan diuji
    cv=3,       # 3-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1   # Gunakan semua core CPU yang tersedia
)

# Fit RandomizedSearchCV pada data latih
random_search.fit(X_train, y_train)

# Tampilkan hyperparameter terbaik
print("Best Parameters:", random_search.best_params_)

# Model terbaik setelah tuning
best_model = random_search.best_estimator_


# model terbaik memiliki parameter yaitu:
# - n_estimators: 500
# - min_samples_split: 2
# - min_samples_leaf: 1
# - max_features: log2
# - max_depth = 10

# ## Data Evaluation

# Menampilkan kinerja model

# In[537]:


importances = best_model.feature_importances_
features = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 
            'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 
            'parking', 'prefarea', 'furnishingstatus']

feature_importance_df = pd.DataFrame({'Features': features, 'Importance': importances})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Features', data=feature_importance_df)
plt.title('Feature Importance')
plt.show()


# In[538]:


y_pred = best_model.predict(X_test)


# In[539]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# In[540]:


print(f"Mean Squared Error: {mse}")
print(f"R-Squared Score: {r2}")


# In[541]:


y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()


# ### Perbandingan dengan linear regression

# In[542]:


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("\nLinear Regression Performance:")
print(f"MSE: {mse_lr}")
print(f"R²: {r2_lr}")

# Perbandingan
print("\nModel Comparison:")
print(f"Random Forest - MSE: {mse}, R²: {r2}")
print(f"Linear Regression - MSE: {mse_lr}, R²: {r2_lr}")


# In[ ]:




