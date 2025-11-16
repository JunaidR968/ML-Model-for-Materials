import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """
    Load data from CSV file and prepare for modeling
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Display basic info about the data
    print("Data Overview:")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    return df

def linear_regression_predict(df, mo_composition):
    """
    Simple linear regression using scipy
    """
    x = df['Mo_Composition'].values
    y = df['Yield Strength'].values
    
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Predict for given composition
    prediction = slope * mo_composition + intercept
    
    return prediction, slope, intercept, r_value**2

def polynomial_regression_predict(df, mo_composition, degree=3):
    """
    Polynomial regression using numpy
    """
    x = df['Mo_Composition'].values
    y = df['Yield Strength'].values
    
    # Fit polynomial
    coefficients = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coefficients)
    
    # Predict for given composition
    prediction = polynomial(mo_composition)
    
    return prediction, coefficients

def weighted_average_predict(df, mo_composition, window=7):
    """
    Weighted average prediction based on distance
    """
    # Sort by Mo_Composition
    sorted_df = df.sort_values('Mo_Composition')
    
    # Find closest values
    distances = np.abs(sorted_df['Mo_Composition'] - mo_composition)
    closest_indices = distances.nsmallest(window).index
    
    # Calculate weights (closer points have higher weight)
    closest_distances = distances.loc[closest_indices]
    weights = 1 / (closest_distances + 0.001)  # Add small value to avoid division by zero
    
    # Weighted average of closest values
    weighted_pred = np.average(sorted_df.loc[closest_indices, 'Yield Strength'], weights=weights)
    
    return weighted_pred

def get_most_accurate_prediction(df, mo_composition):
    """
    Combine multiple models to get the most accurate prediction
    """
    # Get predictions from different models
    linear_pred, slope, intercept, r2_linear = linear_regression_predict(df, mo_composition)
    poly_pred, _ = polynomial_regression_predict(df, mo_composition, degree=3)
    weighted_pred = weighted_average_predict(df, mo_composition, window=7)
    
    # Calculate model confidence scores
    x = df['Mo_Composition'].values
    y = df['Yield Strength'].values
    
    # Linear model confidence (R²)
    linear_confidence = r2_linear
    
    # Polynomial model confidence (using cross-validation-like approach)
    try:
        # Use leave-one-out style validation for polynomial
        poly_errors = []
        for i in range(len(df)):
            temp_df = df.drop(i)
            temp_pred, _ = polynomial_regression_predict(temp_df, df.iloc[i]['Mo_Composition'], degree=3)
            poly_errors.append(abs(temp_pred - df.iloc[i]['Yield Strength']))
        poly_confidence = 1 / (1 + np.mean(poly_errors) / 100)  # Normalize
    except:
        poly_confidence = 0.8
    
    # Weighted average confidence (based on data density around the point)
    distances = np.abs(df['Mo_Composition'] - mo_composition)
    nearby_points = len(distances[distances <= 2])  # Count points within range of 2
    weighted_confidence = min(1.0, nearby_points / 10)  # More nearby points = higher confidence
    
    # Combine predictions with weights based on confidence
    total_confidence = linear_confidence + poly_confidence + weighted_confidence
    if total_confidence > 0:
        final_prediction = (
            linear_pred * linear_confidence +
            poly_pred * poly_confidence +
            weighted_pred * weighted_confidence
        ) / total_confidence
    else:
        # Fallback to weighted average if no confidence
        final_prediction = weighted_pred
    
    # Ensure prediction is within reasonable bounds
    min_strength = df['Yield Strength'].min()
    max_strength = df['Yield Strength'].max()
    final_prediction = max(min_strength, min(max_strength, final_prediction))
    
    return final_prediction, {
        'linear_prediction': linear_pred,
        'polynomial_prediction': poly_pred,
        'weighted_prediction': weighted_pred,
        'final_prediction': final_prediction,
        'confidences': {
            'linear': linear_confidence,
            'polynomial': poly_confidence,
            'weighted': weighted_confidence
        }
    }

def main():
    """
    Main function to run the complete pipeline
    """
    # Define the file path
    file_path = r"C:\Users\Asus\Desktop\MM353_Lab_Session\Titanium_Property_Prediction.csv"
    
    # Load the data
    try:
        print(f"Loading data from: {file_path}")
        df = load_and_prepare_data(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        print("Please check the file path and make sure the file exists.")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    
    # Check for missing values
    if df.isnull().sum().any():
        print("\nMissing values found. Handling missing values...")
        df = df.dropna()
        print(f"Data shape after handling missing values: {df.shape}")
    
    # Check if required columns exist
    required_columns = ['Mo_Composition', 'Yield Strength']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Calculate R² for linear regression for reference
    x = df['Mo_Composition'].values
    y = df['Yield Strength'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value**2
    
    print(f"\n{'='*50}")
    print("MODEL PERFORMANCE")
    print(f"Linear Regression R²: {r_squared:.4f}")
    
    # Example predictions
    print(f"\n{'='*50}")
    print("EXAMPLE PREDICTIONS")
    print("="*40)
    
    test_compositions = [10.0, 15.0, 20.0, 25.0, 30.0]
    
    for comp in test_compositions:
        final_pred, details = get_most_accurate_prediction(df, comp)
        
        print(f"\nMo Composition: {comp}")
        print(f"  Predicted Yield Strength: {final_pred:.2f}")
        print(f"  (Confidences - Linear: {details['confidences']['linear']:.3f}, "
              f"Polynomial: {details['confidences']['polynomial']:.3f}, "
              f"Weighted: {details['confidences']['weighted']:.3f})")
    
    # Interactive prediction
    print(f"\n{'='*50}")
    print("INTERACTIVE PREDICTION")
    print("Enter Mo composition values to get the most accurate prediction")
    print("(type 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\nEnter Mo composition value: ").strip()
            if user_input.lower() == 'quit':
                break
            
            mo_value = float(user_input)
            
            final_pred, details = get_most_accurate_prediction(df, mo_value)
            
            print(f"\n{'='*30}")
            print(f"INPUT: Mo Composition = {mo_value}")
            print(f"OUTPUT: Predicted Yield Strength = {final_pred:.2f}")
            print(f"{'='*30}")
            
            # Optional: Show detailed breakdown
            show_details = input("Show detailed breakdown? (y/n): ").strip().lower()
            if show_details == 'y':
                print(f"\nDetailed Breakdown:")
                print(f"  Linear Regression Prediction: {details['linear_prediction']:.2f}")
                print(f"  Polynomial Regression Prediction: {details['polynomial_prediction']:.2f}")
                print(f"  Weighted Average Prediction: {details['weighted_prediction']:.2f}")
                print(f"  Model Confidences:")
                print(f"    - Linear: {details['confidences']['linear']:.3f}")
                print(f"    - Polynomial: {details['confidences']['polynomial']:.3f}")
                print(f"    - Weighted: {details['confidences']['weighted']:.3f}")
            
        except ValueError:
            print("Please enter a valid number or 'quit' to exit.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()