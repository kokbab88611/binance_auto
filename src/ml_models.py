import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class PredictionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
    
    async def predict_trend(self, df):
        X_train_category, X_test_category, y_train_category, y_test_category = train_test_split(
            df[['Close', 'Volume']], df['Category'],
            test_size=0.2, random_state=42
        )
        df_train_category = pd.concat([X_train_category, y_train_category], axis=1).dropna()
        X_train_category = df_train_category.iloc[:, :-1]
        y_train_category = df_train_category.iloc[:, -1]

        self.model.fit(X_train_category, y_train_category)

        X_new_category = X_test_category.tail(1)
        predicted_category = self.model.predict(X_new_category)
        return predicted_category[0]
