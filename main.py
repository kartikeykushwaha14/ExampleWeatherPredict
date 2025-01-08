import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load dataset
file_path = 'seattle-weather.csv'
data = pd.read_csv(file_path)

# Data preprocessing
data = data.dropna()
data['weather'] = data['weather'].astype('category')
weather_categories = data['weather'].cat.categories  # Get human-readable labels
print(f"Weather categories: {weather_categories}")  # Print out the categories
data['weather'] = data['weather'].cat.codes  # Convert to numeric codes
data['date'] = pd.to_datetime(data['date'])
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data = data.drop(columns=['date'])

# Split data
X = data.drop(columns=['weather'])
y = data['weather']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Create a dictionary to map short labels to full descriptions
weather_mapping = {
    'sun': 'Sunny',  # Mapping 'sun' to 'Sunny'
    'rain': 'Rainy',  # Example mapping, if 'rain' is used in the dataset
    'cloud': 'Cloudy',  # Example mapping, if 'cloud' is used in the dataset
    'drizzle': 'Light Rain',  # Mapping 'drizzle' to 'Light Rain'
    'fog': 'Foggy',  # Mapping 'fog' to 'Foggy'
    'snow': 'Snowy',  # Mapping 'snow' to 'Snowy'
    # Add other mappings if necessary
}


# Function to predict weather based on user input
def predict_weather():
    print("Enter the weather details for prediction:")

    # Get user input for all features
    precipitation = float(input("Precipitation (mm): "))
    temp_max = float(input("Maximum Temperature (°C): "))
    temp_min = float(input("Minimum Temperature (°C): "))
    wind = float(input("Wind Speed (km/h): "))
    month = int(input("Month (1-12): "))
    day = int(input("Day (1-31): "))

    # Create a dataframe from the user input
    user_data = {
        'precipitation': [precipitation],
        'temp_max': [temp_max],
        'temp_min': [temp_min],
        'wind': [wind],
        'month': [month],
        'day': [day]
    }
    user_df = pd.DataFrame(user_data)

    # Predict weather
    weather_code = model.predict(user_df)[0]  # Get the predicted numeric code
    predicted_weather = weather_categories[weather_code]  # Map the numeric code back to the category

    # Use the weather_mapping to translate to full description
    full_weather_description = weather_mapping.get(predicted_weather, predicted_weather)  # Default to original if not mapped
    print(f"Predicted weather for today: {full_weather_description}")

    # Visualization using Matplotlib
    labels = list(weather_mapping.values())
    sizes = [1 if label == full_weather_description else 0 for label in labels]

    plt.figure(figsize=(8, 6))
    plt.bar(labels, sizes, color=['blue' if label == full_weather_description else 'gray' for label in labels])
    plt.title("Predicted Weather")
    plt.ylabel("Prediction Confidence")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Call the function to get input and predict weather
predict_weather()
