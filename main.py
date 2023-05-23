import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def load_data(file_path):
    """
    Function to load the dataset from a CSV file.
    """
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    """
    Function to preprocess the data.
    """
    # Handle missing values (if any)
    # ...

    # Handle outliers (if any)
    # ...

    # Perform feature scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    return scaled_data


def perform_feature_engineering(data, target_variable, num_features):
    """
    Function to perform feature engineering.
    """
    # Perform dimensionality reduction using PCA
    pca = PCA(n_components=num_features)
    reduced_data = pca.fit_transform(data)

    # Perform feature selection using SelectKBest
    selector = SelectKBest(f_classif, k=num_features)
    selected_data = selector.fit_transform(data, target_variable)

    return reduced_data, selected_data


def train_model(data, target_variable):
    """
    Function to train the model.
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, target_variable, test_size=0.2, random_state=42)

    # Train the model (e.g., RandomForestClassifier)
    model = RandomForestClassifier()

    # Define hyperparameters for tuning (if needed)
    param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}

    # Perform hyperparameter tuning using GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    return best_model


def evaluate_model(model, data, target_variable):
    """
    Function to evaluate the model.
    """
    # Predict the target variable
    predictions = model.predict(data)

    # Calculate accuracy score
    accuracy = accuracy_score(target_variable, predictions)

    return accuracy


# Main function
def main():
    # Load the dataset
    data = load_data('data.csv')

    # Preprocess the data
    preprocessed_data = preprocess_data(data)

    # Perform feature engineering
    reduced_data, selected_data = perform_feature_engineering(preprocessed_data, data['target'], num_features=10)

    # Train and evaluate the model using reduced data
    model_reduced = train_model(reduced_data, data['target'])
    accuracy_reduced = evaluate_model(model_reduced, reduced_data, data['target'])
    print(f"Accuracy using reduced data: {accuracy_reduced}")

    # Train and evaluate the model using selected data
    model_selected = train_model(selected_data, data['target'])
    accuracy_selected = evaluate_model(model_selected, selected_data, data['target'])
    print(f"Accuracy using selected data: {accuracy_selected}")


if __name__ == '__main__':
    main()
