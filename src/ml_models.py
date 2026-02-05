import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class PredictionModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
    
    def predict_trend(self, df):
        """
        Predicts whether the next trend is 'bullish' or 'bearish' 
        using Logistic Regression on Close price and Volume.
        """
        # Prepare data
        df_subset = df[['Close', 'Volume', 'Category']].dropna()
        
        if len(df_subset) < 20: # Need enough data points
            return None
            
        X = df_subset[['Close', 'Volume']]
        y = df_subset['Category']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predict on the latest data point available in the test set 
        # (In a real scenario, this should be the live current candle)
        X_new = X_test.tail(1) 
        
        if X_new.empty:
            return None
            
        prediction = self.model.predict(X_new)
        return prediction[0]
