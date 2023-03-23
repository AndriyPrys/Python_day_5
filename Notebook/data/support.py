# Day 5 - Feature Engineering & Selection


#Exercise 1 - Label encoding
def codice_5_1():
    print('''
    df_credit=pd.DataFrame()
    le = preprocessing.LabelEncoder()
    df_credit['credit_scikit']=le.fit_transform(df['credit_lev'])
    df_credit['credit_pandas']=df["credit_lev"].astype('category').cat.codes
    df_credit
    ''')

#Exercise 2 - One hot encoding
def codice_5_2():
    print('''
    df_marriage=pd.DataFrame()
    df_marriage['marriage_married']=np.where(df["marriage"]=='married', 1, 0)
    df_marriage['marriage_others']=np.where(df["marriage"]=='others', 1, 0)
    df_marriage['marriage_single']=np.where(df["marriage"]=='single', 1, 0)
    df_marriage['marriage_unknown']=np.where(df["marriage"]=='unknown', 1, 0)
    df_marriage
    ''')

#Exercise 3 - Zscore
def codice_5_3():
    print('''
    z_age = stats.zscore(df[['age']])
    z_age
    ''')
    
def check_5_3(solution):
    try:
        if round(solution,3)==-1.246:
            return ("Correct answer!")
        else:
            return ("Retry!") 
    except:
         return ("Retry!")   

    
#Exercise 4 - Feature scaling
def codice_5_4():
    print('''
    plt.figure(figsize=(12,5))
    ax1 = plt.subplot(1, 2, 1)
    ax1=sns.boxplot(y=df.age_std)
    ax2 = plt.subplot(1, 2, 2, sharey=ax1)
    ax2=sns.boxplot(y=df.age_norm)
    ''')

#Exercise 5 - PCA
def codice_5_5():
    print('''
    plt.figure(figsize=(10,5))
    df_pca_plot = pd.concat([principal_df.iloc[:,:5], pd.Series(y_pca, name = 'Target')],axis=1)
    sns.scatterplot(data=df_pca_plot, x='PC_1', y='PC_2', hue="Target")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.show()
    ''')
    
# Exercise 6 - House prices

def codice_5_6():
    print('''
    #6.1
    X_house=df_house.select_dtypes(include=['number']).drop(['Id','SalePrice'],axis=1)
    
    #6.2
    X_house=X_house.fillna(X_house.mean())
    
    #6.3
    def random_forest_feature_ranking(forest,X,y):    
        forest.fit(X, y)

        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        for f in range(X.shape[1]):
            print("%d. feature %d %s (%f)" % (f + 1, indices[f], X.columns[indices[f]], importances[indices[f]]))

        # Plot the feature importances of the forest
        plt.figure(figsize=(10,4))
        plt.title("Feature importances")
        plt.bar(range(X.shape[1]), importances[indices],
               color="r", yerr=std[indices], align="center")
        plt.xticks(range(X.shape[1]), X.columns[indices],rotation=90)
        plt.xlim([-1, X.shape[1]])
        plt.show()

    y_house=df_house['SalePrice']
    random_forest_feature_ranking(forest,X_house,y_house)
    
    #6.4
    rfe_house = RFE(estimator=LinearRegression(), n_features_to_select=8)
    rfe_house = rfe_house.fit(X_house, y_house)
    print(X_house.columns[rfe_house.support_])
    
    #6.5
    plt.figure(figsize=(25,20))
    sns.heatmap(df_house.corr(),annot=True)
    ''')
    
def check_5_6_3(solution):
    try:
        if solution=="LotArea":
            return ("Correct answer!")
        else:
            return ("Retry!")    
    except:
         return ("Retry!")   
    
def check_5_6_5(solution):
    try:
        if solution=="OverallQual":
            return ("Correct answer!")
        else:
            return ("Retry!")
    except:
         return ("Retry!")   
    
# Exercise 7 - Ames   
    
def codice_5_7_1():
    print('''    

df["LivLotRatio"] = df.GrLivArea / df.LotArea
df["Spaciousness"] = (df.FirstFlrSF + df.SecondFlrSF) / df.TotRmsAbvGrd
df["TotalOutsideSF"] = df.WoodDeckSF + df.OpenPorchSF + df.EnclosedPorch + df.Threeseasonporch + df.ScreenPorch

''')        
    
def codice_5_7_2():
    print('''    

# Basic Version
df_dummies = pd.get_dummies(df.BldgType, prefix="Bldg")
df_inter = pd.DataFrame([df_dummies[c]*df.GrLivArea for c in df_dummies.columns], index = df_dummies.columns).T

# Advanced Version
df_dummies = pd.get_dummies(df.BldgType, prefix="Bldg")
df_inter = df_dummies.mul(df.GrLivArea, axis=0)

''')     
    
def codice_5_7_3():
    print('''    
 
# Basic Version
df["PorchTypes"] = (df[[
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "Threeseasonporch",
    "ScreenPorch",
]] > 0.0).sum(axis=1)
     
# Advanced Version   
df["PorchTypes"] = df[[
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "Threeseasonporch",
    "ScreenPorch",
]].gt(0.0).sum(axis=1)
  
''')     
         
def codice_5_7_4():
    print('''    
 
# Basic Version
df["MSClass"] = pd.Series([c.split("_")[0] for c in df.MSSubClass])
     
# Advanced Version   
df["MSClass"] = df.MSSubClass.str.split("_", n=1, expand=True)[0]
  
''')     

        
def codice_5_7_5():
    print('''    
 
# Basic Version
median_groups = df.groupby("Neighborhood").GrLivArea.median()
median_groups = median_groups.reset_index().rename(columns = {'GrLivArea':'MedNhbdArea'})
df = df.merge(median_groups, on = 'Neighborhood', how = 'left')     
     
# Advanced Version   
df["MedNhbdArea"] = df.groupby("Neighborhood")["GrLivArea"].transform("median")

''')          
        
        