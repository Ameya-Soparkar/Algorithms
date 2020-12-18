class Regression():
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def Linear_Regression(self):
        from sklearn.linear_model import LinearRegression
        model_linearreg = LinearRegression()   
        model_linearreg.fit(self.X_train,self.y_train)
        return model_linearreg.score(self.X_test,self.y_test)

    def Support_Vector_Regression(self):
        from sklearn.svm import SVR
        svr_model = SVR(kernel='rbf')
        svr_model.fit(self.X_train, self.y_train)
        return svr_model.score(self.X_test,self.y_test)

    def Decision_Tree(self):
        from sklearn.tree import DecisionTreeRegressor
        decisiontree_model = DecisionTreeRegressor()
        decisiontree_model.fit(self.X_train, self.y_train)
        return decisiontree_model.score(self.X_test,self.y_test)

    def Randowm_Forest(self):
        from sklearn.ensemble import RandomForestClassifier
        randomforest_model = RandomForestClassifier(n_estimators=10, random_state=0)
        randomforest_model.fit(self.X_train, self.y_train)
        return randomforest_model.score(self.X_test,self.y_test)


    def knowall_score(self):
        r= Regression(self.X_train,self.X_test,self.y_train,self.y_test)
        print(f'(The score of Linear Regression Model is {r.Linear_Regression()})')
        #print(f'(The score of Support Vector Regression is {r.Support_Vector_Regression()})')
        print(f'(The score of Decision tree regression is {r.Decision_Tree()})')
        print(f'(The score of Random Forest regression is {r.Randowm_Forest()})')

``


        



