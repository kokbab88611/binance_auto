from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(data_collector):
    # Prepare the dataset
    dataset = data_collector.prepare_dataset()
    X = dataset.drop('target', axis=1)
    y = dataset['target']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a machine learning model (e.g., Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model to a file
    joblib.dump(model, 'trading_model.pkl')

    print(f"Model training completed. Test accuracy: {model.score(X_test, y_test)}")
    return model
