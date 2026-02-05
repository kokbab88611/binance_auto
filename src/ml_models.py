import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class TrendPredictor:
    def __init__(self):
        # Starting with Logistic Regression for baseline
        # TODO: Experiment with XGBoost for better non-linear capture
        # TODO: Add hyperparameter tuning grid
        self.model = LogisticRegression(max_iter=1000)
    
    def predict(self, df):
        """
        Binary classification of trend (bullish/bearish).
        Uses simple features (Close, Volume) for now.
        """
        # Need at least 20 candles to fit meaningful trend
        subset = df[['Close', 'Volume', 'Category']].dropna()
        if len(subset) < 20: 
            return None
            
        X = subset[['Close', 'Volume']]
        y = subset['Category']
        
        # Standard train/test split to validate accuracy before live inference
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False # Don't shuffle time series!
        )
        
        self.model.fit(X_train, y_train)
        
        # Predict on the live candle (last row)
        current_features = X_test.tail(1) 
        
        if current_features.empty:
            return None
            
        prediction = self.model.predict(current_features)
        return prediction[0]
