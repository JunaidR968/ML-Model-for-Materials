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
    
    return prediction, coefficients, polynomial

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

def exponential_model_predict(df, mo_composition):
    """
    Exponential model prediction
    """
    x = df['Mo_Composition'].values
    y = df['Yield Strength'].values
    
    # Fit exponential model: y = a * exp(b * x)
    try:
        # Use linear regression on log(y) vs x
        log_y = np.log(y)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_y)
        a = np.exp(intercept)
        b = slope
        prediction = a * np.exp(b * mo_composition)
        return prediction, a, b, r_value**2
    except:
        # Fallback to linear if exponential fails
        linear_pred, _, _, _ = linear_regression_predict(df, mo_composition)
        return linear_pred, 0, 0, 0

def power_model_predict(df, mo_composition):
    """
    Power law model prediction
    """
    x = df['Mo_Composition'].values
    y = df['Yield Strength'].values
    
    # Fit power model: y = a * x^b
    try:
        # Use linear regression on log(y) vs log(x)
        # Filter out non-positive x values
        positive_mask = x > 0
        if np.sum(positive_mask) < 10:  # Need sufficient positive points
            raise ValueError("Insufficient positive x values")
            
        x_positive = x[positive_mask]
        y_positive = y[positive_mask]
        
        log_x = np.log(x_positive)
        log_y = np.log(y_positive)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
        a = np.exp(intercept)
        b = slope
        if mo_composition > 0:
            prediction = a * (mo_composition ** b)
        else:
            prediction = a * (0.1 ** b)  # Fallback for non-positive inputs
        return prediction, a, b, r_value**2
    except:
        # Fallback to linear
        linear_pred, _, _, _ = linear_regression_predict(df, mo_composition)
        return linear_pred, 0, 0, 0

def get_most_accurate_prediction(df, mo_composition):
    """
    Combine multiple models to get the most accurate prediction
    """
    # Get predictions from different models
    linear_pred, slope, intercept, r2_linear = linear_regression_predict(df, mo_composition)
    poly_pred, _, poly_func = polynomial_regression_predict(df, mo_composition, degree=3)
    weighted_pred = weighted_average_predict(df, mo_composition, window=7)
    exp_pred, exp_a, exp_b, r2_exp = exponential_model_predict(df, mo_composition)
    power_pred, power_a, power_b, r2_power = power_model_predict(df, mo_composition)
    
    # Calculate model confidence scores
    x = df['Mo_Composition'].values
    y = df['Yield Strength'].values
    
    # Linear model confidence (R²)
    linear_confidence = r2_linear
    
    # Polynomial model confidence
    try:
        poly_errors = []
        for i in range(len(df)):
            temp_df = df.drop(i)
            temp_pred, _, _ = polynomial_regression_predict(temp_df, df.iloc[i]['Mo_Composition'], degree=3)
            poly_errors.append(abs(temp_pred - df.iloc[i]['Yield Strength']))
        poly_confidence = 1 / (1 + np.mean(poly_errors) / 100)
    except:
        poly_confidence = 0.7
    
    # Weighted average confidence
    distances = np.abs(df['Mo_Composition'] - mo_composition)
    nearby_points = len(distances[distances <= 2])
    weighted_confidence = min(1.0, nearby_points / 10)
    
    # Exponential model confidence
    exp_confidence = r2_exp if r2_exp > 0 else 0.6
    
    # Power model confidence
    power_confidence = r2_power if r2_power > 0 else 0.5
    
    # Combine predictions with weights based on confidence
    confidences = [linear_confidence, poly_confidence, weighted_confidence, exp_confidence, power_confidence]
    predictions = [linear_pred, poly_pred, weighted_pred, exp_pred, power_pred]
    
    total_confidence = sum(confidences)
    if total_confidence > 0:
        final_prediction = sum(p * c for p, c in zip(predictions, confidences)) / total_confidence
    else:
        final_prediction = weighted_pred
    
    # Ensure prediction is within reasonable bounds
    min_strength = df['Yield Strength'].min()
    max_strength = df['Yield Strength'].max()
    final_prediction = max(min_strength, min(max_strength, final_prediction))
    
    return final_prediction, {
        'linear_prediction': linear_pred,
        'polynomial_prediction': poly_pred,
        'weighted_prediction': weighted_pred,
        'exponential_prediction': exp_pred,
        'power_prediction': power_pred,
        'final_prediction': final_prediction,
        'confidences': {
            'linear': linear_confidence,
            'polynomial': poly_confidence,
            'weighted': weighted_confidence,
            'exponential': exp_confidence,
            'power': power_confidence
        },
        'models': {
            'linear': (slope, intercept),
            'polynomial': poly_func,
            'exponential': (exp_a, exp_b),
            'power': (power_a, power_b)
        }
    }

def plot_all_models(df, model_details):
    """
    Create comprehensive plots of all ML models
    """
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Generate x range for smooth curves
    x_range = np.linspace(df['Mo_Composition'].min(), df['Mo_Composition'].max(), 200)
    
    # Plot 1: All Models Comparison
    plt.subplot(2, 3, 1)
    plt.scatter(df['Mo_Composition'], df['Yield Strength'], alpha=0.6, color='black', label='Actual Data', s=20)
    
    # Plot each model
    # Linear
    linear_preds = [model_details['models']['linear'][0] * x + model_details['models']['linear'][1] for x in x_range]
    plt.plot(x_range, linear_preds, 'r-', linewidth=2, label=f'Linear (R²: {model_details["confidences"]["linear"]:.3f})')
    
    # Polynomial
    poly_preds = [model_details['models']['polynomial'](x) for x in x_range]
    plt.plot(x_range, poly_preds, 'g-', linewidth=2, label=f'Polynomial (Conf: {model_details["confidences"]["polynomial"]:.3f})')
    
    # Exponential
    exp_preds = [model_details['models']['exponential'][0] * np.exp(model_details['models']['exponential'][1] * x) for x in x_range]
    plt.plot(x_range, exp_preds, 'b-', linewidth=2, label=f'Exponential (R²: {model_details["confidences"]["exponential"]:.3f})')
    
    # Power
    power_preds = [model_details['models']['power'][0] * (max(0.1, x) ** model_details['models']['power'][1]) for x in x_range]
    plt.plot(x_range, power_preds, 'm-', linewidth=2, label=f'Power (R²: {model_details["confidences"]["power"]:.3f})')
    
    plt.xlabel('Mo Composition')
    plt.ylabel('Yield Strength')
    plt.title('All ML Models Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Individual Model Plots
    models_info = [
        ('Linear Regression', linear_preds, 'red', model_details["confidences"]["linear"]),
        ('Polynomial Regression', poly_preds, 'green', model_details["confidences"]["polynomial"]),
        ('Exponential Model', exp_preds, 'blue', model_details["confidences"]["exponential"]),
        ('Power Law Model', power_preds, 'magenta', model_details["confidences"]["power"])
    ]
    
    for i, (name, preds, color, confidence) in enumerate(models_info, 2):
        plt.subplot(2, 3, i)
        plt.scatter(df['Mo_Composition'], df['Yield Strength'], alpha=0.6, color='gray', s=15)
        plt.plot(x_range, preds, color=color, linewidth=2.5)
        plt.xlabel('Mo Composition')
        plt.ylabel('Yield Strength')
        plt.title(f'{name}\nConfidence: {confidence:.3f}')
        plt.grid(True, alpha=0.3)
    
    # Plot 3: Model Confidence Comparison
    plt.subplot(2, 3, 6)
    model_names = ['Linear', 'Polynomial', 'Exponential', 'Power']
    confidences = [model_details['confidences']['linear'], 
                   model_details['confidences']['polynomial'],
                   model_details['confidences']['exponential'],
                   model_details['confidences']['power']]
    
    colors = ['red', 'green', 'blue', 'magenta']
    bars = plt.bar(model_names, confidences, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, confidence in zip(bars, confidences):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{confidence:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Confidence Score')
    plt.title('Model Confidence Comparison')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Additional Plot: Residual Analysis
    fig2, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Calculate residuals for each model
    models_calc = [
        ('Linear', lambda x: model_details['models']['linear'][0] * x + model_details['models']['linear'][1]),
        ('Polynomial', model_details['models']['polynomial']),
        ('Exponential', lambda x: model_details['models']['exponential'][0] * np.exp(model_details['models']['exponential'][1] * x)),
        ('Power', lambda x: model_details['models']['power'][0] * (max(0.1, x) ** model_details['models']['power'][1]))
    ]
    
    for idx, (name, model_func) in enumerate(models_calc):
        row = idx // 2
        col = idx % 2
        
        predictions = [model_func(x) for x in df['Mo_Composition']]
        residuals = df['Yield Strength'] - predictions
        
        axes[row, col].scatter(predictions, residuals, alpha=0.6, color=colors[idx])
        axes[row, col].axhline(y=0, color='red', linestyle='--')
        axes[row, col].set_xlabel('Predicted Values')
        axes[row, col].set_ylabel('Residuals')
        axes[row, col].set_title(f'{name} Model - Residual Plot')
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

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
    
    # Generate model details for plotting (using a sample point)
    sample_comp = 15.0
    final_pred, model_details = get_most_accurate_prediction(df, sample_comp)
    
    # Plot all models
    print(f"\n{'='*50}")
    print("GENERATING MODEL PLOTS...")
    plot_all_models(df, model_details)
    
    # Example predictions
    print(f"\n{'='*50}")
    print("EXAMPLE PREDICTIONS")
    print("="*40)
    
    test_compositions = [10.0, 15.0, 20.0, 25.0, 30.0]
    
    for comp in test_compositions:
        final_pred, details = get_most_accurate_prediction(df, comp)
        
        print(f"\nMo Composition: {comp}")
        print(f"  Predicted Yield Strength: {final_pred:.2f}")
        print(f"  Model Confidences:")
        print(f"    - Linear: {details['confidences']['linear']:.3f}")
        print(f"    - Polynomial: {details['confidences']['polynomial']:.3f}")
        print(f"    - Exponential: {details['confidences']['exponential']:.3f}")
        print(f"    - Power: {details['confidences']['power']:.3f}")
        print(f"    - Weighted: {details['confidences']['weighted']:.3f}")
    
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
            
        except ValueError:
            print("Please enter a valid number or 'quit' to exit.")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()