#data manipulation
import numpy as np
import pandas as pd

#data preprocessing
import missingno as msno
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest,f_regression,chi2
from sklearn.preprocessing import LabelEncoder

#model building
import xgboost as xgt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#model evaulation
from sklearn.metrics import r2_score

data = pd.read_csv('train.csv')
data.dtypes.value_counts()
data.head()
df_int = data.select_dtypes(include=['int64', 'float64'])


#DEALING WITH MISING VALUES
df_int.isna().mean()*100

msno.matrix(df_int)

df_int.isna().sum()

#using imputation methods to fill in the missing data
imp =SimpleImputer(strategy='mean')
m_imp = df_int.copy()
m_imp.loc[:,:] = imp.fit_transform(m_imp.loc[:,:])

imp_2 = KNNImputer(n_neighbors = 2)
knn_imp = df_int.copy()
knn_imp.loc[:,:]=imp_2.fit_transform(knn_imp.loc[:,:])

imputations = {'mean imputations': m_imp,
             'knn imputations': knn_imp}

#visualizing how imputed data resembles original data
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(15,8))
for ax, key in zip(ax.flatten(), imputations):
    imputations[key].hist("LotFrontage", bins=30, ax=ax, alpha=1)
    df_int.hist("LotFrontage",bins=30, ax=ax)
    ax.set_title(key)
    plt.tight_layout()


# The orange histograms represent the original data while the blue histograms represent the imputed data. From the graphs ot can be noticed that the missing data that was imputd using k-nearest neighbours followed the pattern of the original data unlike that using mean imputations.

df_cat = data.select_dtypes(include='object')

df_cat.isna().mean()*100
df_cat.isna().sum()

msno.matrix(df_cat)

df_cat.drop(columns=['Alley', 'PoolQC', 'Fence','MiscFeature','FireplaceQu'], inplace=True)
df_cat['Electrical'] = df_cat['Electrical'].fillna(df_cat['Electrical'].mode()[0])
df_cat['MasVnrType'] = df_cat['MasVnrType'].fillna(df_cat['MasVnrType'].mode()[0])

#using a unique value to represent the remaining missing values to maintain the originality of the data
df_cat = df_cat.fillna('Unknown')

#using a labelencoder to transform categorical variables into numerical variables for model
copy = df_cat.copy()
enc = LabelEncoder()
for col in copy.columns:
    copy[col] = enc.fit_transform(copy[col])


# # MODEL BUILDING WITHOUT FEATURE SELECTION
X,y = knn_imp.loc[:,df_int.columns!='SalePrice'],knn_imp.loc[:,'SalePrice']

new_d = pd.concat([X, copy], axis=1)

#splitting the data into training and validation for model building and evaluation  
X_train, X_test, y_train, y_test = train_test_split(new_d, y, test_size=0.3, random_state=42)

model = xgt.XGBRegressor(random_state=42)
model.fit(X_train, y_train)

m = model.predict(X_test)
r2_score(y_test,m)


model2 = RandomForestRegressor()
model2.fit(X_train, y_train)

n = model2.predict(X_test)
r2_score(y_test, n)


# # MODEL BUILDING WITH FEATURE SELECTION

# selecting features for numerical variables
selector =  SelectKBest(f_regression,k='all')
X_int =selector.fit_transform(X,y)

names = X.columns.values[selector.get_support()]
scores = selector.scores_[selector.get_support()]
names_scores = list(zip(names, scores))
ns_df = pd.DataFrame(data = names_scores, columns=['Feat_names', 'F_Scores'])
ns_df_sorted = ns_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])
print(ns_df_sorted)

#removing nunmerical variables for model buidling with F_score<10
selector3 = SelectKBest(f_regression, k=20)
X2 = selector3.fit_transform(X,y)
names3 = X.columns.values[selector3.get_support()]
fea_int = X.loc[:,names3]
fea_int.shape


# FEATURE SELECTION FOR CATEGORICAL VARIABLES

selector2 =  SelectKBest(chi2,k='all')
X_cat =selector2.fit_transform(copy,y)

names2 = copy.columns.values[selector2.get_support()]
scores2 = selector2.scores_[selector2.get_support()]
names_scores2 = list(zip(names2, scores2))
ns_df2 = pd.DataFrame(data = names_scores2, columns=['Feat_names', 'F_Scores'])
ns_df_sorted2 = ns_df2.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])
print(ns_df_sorted2)


#removing variables with F-score <10
selector4 =  SelectKBest(chi2,k=37)
X_cat =selector4.fit_transform(copy,y)
names4 = copy.columns.values[selector4.get_support()]
fea_cat = copy.loc[:,names4]
fea_cat.shape

m_data = pd.concat([fea_int, fea_cat], axis=1)

X_train1, X_test1, y_train1, y_test1 = train_test_split(m_data, y, test_size=0.3, random_state=42)


model3 = xgt.XGBRegressor(objective='reg:linear', random_state=42)
model3.fit(X_train1, y_train1)

pred3 = model3.predict(X_test1)
r2_score(y_test1,pred3)


model4 = RandomForestRegressor()
model4.fit(X_train1, y_train1)

pred4 = model4.predict(X_test1)
r2_score(y_test1, pred4)


# random forest regressor with selected features produced best outcome as such we would use it for prediction

# # PREDICTIONS
test = pd.read_csv('test.csv')

test.dtypes.value_counts()
test_int = test.select_dtypes(include=['int64', 'float64'])
test_cat = test.select_dtypes(include='object')

test_int.isnull().sum()

test_int['BsmtFinSF1'] = test_int['BsmtFinSF1'].fillna(test_int['BsmtFinSF1'].mode()[0])
test_int['BsmtFinSF2'] = test_int['BsmtFinSF2'].fillna(test_int['BsmtFinSF2'].mode()[0])
test_int['BsmtUnfSF'] = test_int['BsmtUnfSF'].fillna(test_int['BsmtUnfSF'].mode()[0])
test_int['TotalBsmtSF'] = test_int['TotalBsmtSF'].fillna(test_int['TotalBsmtSF'].mode()[0])
test_int['BsmtHalfBath'] = test_int['BsmtHalfBath'].fillna(test_int['BsmtHalfBath'].mode()[0])
test_int['BsmtFullBath'] = test_int['BsmtFullBath'].fillna(test_int['BsmtFullBath'].mode()[0])
test_int['GarageCars'] = test_int['GarageCars'].fillna(test_int['GarageCars'].mode()[0])
test_int['GarageArea'] = test_int['GarageArea'].fillna(test_int['GarageArea'].mode()[0])

test_int.loc[:,:] = imp_2.fit_transform(test_int.loc[:,:])

test_int.drop(columns=['Id', 'MSSubClass','OverallCond','BsmtFinSF2','BsmtUnfSF','LowQualFinSF','BsmtFullBath', 'BsmtHalfBath','BedroomAbvGr', 'KitchenAbvGr','EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea','MiscVal', 'MoSold', 'YrSold'],inplace=True)

test_cat.isnull().sum()
test_cat.drop(columns=['Alley', 'PoolQC', 'Fence','MiscFeature','FireplaceQu'], inplace=True)
test_cat['MSZoning'] = test_cat['MSZoning'].fillna(test_cat['MSZoning'].mode()[0])
test_cat['Utilities'] = test_cat['Utilities'].fillna(test_cat['Utilities'].mode()[0])
test_cat['Exterior1st'] = test_cat['Exterior1st'].fillna(test_cat['Exterior1st'].mode()[0])
test_cat['Exterior2nd'] = test_cat['Exterior2nd'].fillna(test_cat['Exterior2nd'].mode()[0])
test_cat['KitchenQual'] = test_cat['KitchenQual'].fillna(test_cat['KitchenQual'].mode()[0])
test_cat['Functional'] = test_cat['Functional'].fillna(test_cat['Functional'].mode()[0])
test_cat['SaleType'] = test_cat['SaleType'].fillna(test_cat['SaleType'].mode()[0])

test_cat = test_cat.fillna('Unknown')

for col in test_cat.columns:
    test_cat[col] = enc.fit_transform(test_cat[col])

test_cat.drop(columns='Street', inplace=True)

test_data = pd.concat([test_int,test_cat], axis = 1)

test_data.shape

predicted = model4.predict(test_data)

predicted = pd.DataFrame(predicted)
predicted = predicted.rename({0:'SalePrice'},axis=1)
predicted['Id'] = test['Id']
predicted = predicted[['Id','SalePrice']]
predicted
