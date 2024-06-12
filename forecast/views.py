from django.shortcuts import render
import pandas as pd
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import re
import missingno as mso
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import threading
import io
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
from django.shortcuts import render
from weather.forms import WeatherPredictionForm

data = pd.read_csv("C:\\Users\\12082\\Downloads\\seattle-weather.csv")


def head(request):
    data = pd.read_csv("C:\\Users\\12082\\Downloads\\seattle-weather.csv")
    data=data.head(10)
    # Convert DataFrame to HTML table
    table_html = data.to_html(classes='table table-striped', index=True)
    context = {'table_html': table_html}
    return render(request, 'template_name.html', context)
def plot(request):
    warnings.filterwarnings('ignore')
    sns.set(style="whitegrid")

    # Define your data (assuming it's defined somewhere in your code)
    data = pd.read_csv("C:\\Users\\12082\\Downloads\\seattle-weather.csv")

    # Create the count plot
    plt.figure(figsize=(8, 6))
    sns.countplot(x="weather", data=data, palette="hls")
    plt.title("Weather Count Plot")
    plt.xlabel("Weather")
    plt.ylabel("Count")

    # Save the plot to a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Convert the plot to base64 for embedding in HTML
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plot_url = f'data:image/png;base64,{plot_data}'

    # Render the template with the plot URL
    return render(request, 'plot.html', {'plot_image_path': plot_url})
def generate_plots(request):
    warnings.filterwarnings('ignore')
    sns.set(style="darkgrid")
    data = pd.read_csv("C:\\Users\\12082\\Downloads\\seattle-weather.csv")
    fig,axs=plt.subplots(2,2,figsize=(10,8))
    sns.histplot(data=data,x="precipitation",kde=True,ax=axs[0,0],color='green')
    sns.histplot(data=data,x="temp_max",kde=True,ax=axs[0,1],color='red')
    sns.histplot(data=data,x="temp_min",kde=True,ax=axs[1,0],color='skyblue')
    sns.histplot(data=data,x="wind",kde=True,ax=axs[1,1],color='orange')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plot_url = f'data:histimage/png;base64,{plot_data}'
    return render(request, 'histogram.html', {'plot_image_path': plot_url})


def violin(request):

    sns.set(style="darkgrid")

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Violin plots
    sns.violinplot(data=data, x="precipitation", ax=axs[0, 0], color='green')
    sns.violinplot(data=data, x="temp_max", ax=axs[0, 1], color='red')
    sns.violinplot(data=data, x="temp_min", ax=axs[1, 0], color='skyblue')
    sns.violinplot(data=data, x="wind", ax=axs[1, 1], color='yellow')

    # KDE plots
    sns.kdeplot(data=data["precipitation"], ax=axs[0, 0], color='green', shade=True)
    sns.kdeplot(data=data["temp_max"], ax=axs[0, 1], color='red', shade=True)
    sns.kdeplot(data=data["temp_min"], ax=axs[1, 0], color='skyblue', shade=True)
    sns.kdeplot(data=data["wind"], ax=axs[1, 1], color='yellow', shade=True)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plot_url = f'data:violinimage/png;base64,{plot_data}'
    return render(request, 'violin.html', {'plot_image_path': plot_url})


def boxplot(request):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    # Create a boxplot
    sns.boxplot(x="precipitation", y="weather", data=data, palette="YlOrBr")
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plot_url = f'data:boximage/png;base64,{plot_data}'
    return render(request, 'box.html', {'plot_image_path': plot_url})

def boxplott(request):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    # Create a boxplot
    sns.boxplot(x="temp_max", y="weather", data=data, palette="inferno")

    # Set plot labels
    plt.xlabel("Maximum Temperature (°C)")
    plt.ylabel("Weather")
    plt.title("Box Plot of Maximum Temperature by Weather")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plot_url = f'data:boxpimage/png;base64,{plot_data}'
    return render(request, 'boxp.html', {'plot_image_path': plot_url})

def boxplotwind(request):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    # Create a boxplot
    sns.boxplot(x="wind", y="weather", data=data, palette="inferno")

    # Set plot labels
    plt.xlabel("Wind Speed")
    plt.ylabel("Weather")
    plt.title("Box Plot of Wind Speed by Weather")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plot_url = f'data:boxpwimage/png;base64,{plot_data}'
    return render(request, 'boxpw.html', {'plot_image_path': plot_url})

def minimumtemp(request):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    # Create a boxplot
    sns.boxplot(x="temp_min", y="weather", data=data, palette="YlOrBr")

    # Set plot labels
    plt.xlabel("Minimum Temperature (°C)")
    plt.ylabel("Weather")
    plt.title("Box Plot of Minimum Temperature by Weather")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plot_url = f'data:minimage/png;base64,{plot_data}'
    return render(request, 'min.html', {'plot_image_path': plot_url})

def corelation(request):
    # Assuming your DataFrame contains a non-numeric column named "date_column"
    # Drop non-numeric columns before computing correlations
    numerical_data = data.select_dtypes(include=['number'])

    # Compute correlations for numerical columns
    correlation_matrix = numerical_data.corr()

    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(12, 7))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plot_url = f'data:corelation/png;base64,{plot_data}'
    return render(request, 'corelation.html', {'plot_image_path': plot_url})
    
def percentage_of_weather_condition(request):
    import pandas as pd

    # Assuming you have already loaded your data into a DataFrame named 'data'
    data = pd.read_csv("C:\\Users\\12082\\Downloads\\seattle-weather.csv")

    # Count the occurrences of each weather condition
    countrain = len(data[data.weather == "rain"])
    countsun = len(data[data.weather == "sun"])
    countdrizzle = len(data[data.weather == "drizzle"])
    countsnow = len(data[data.weather == "snow"])
    countfog = len(data[data.weather == "fog"])

    # Calculate the percentages
    total_entries = len(data)
    percent_rain = (countrain / total_entries) * 100
    percent_sun = (countsun / total_entries) * 100
    percent_drizzle = (countdrizzle / total_entries) * 100
    percent_snow = (countsnow / total_entries) * 100
    percent_fog = (countfog / total_entries) * 100
    context = {'rain':percent_rain,'sun':percent_sun,'drizzle':percent_drizzle,'snow':percent_snow,'fog':percent_fog}
   
    return render(request,'percentage.html',context)

def Precipitation(request):
    # Read the data (assuming it's defined somewhere in your code)
    data = pd.read_csv("C:\\Users\\12082\\Downloads\\seattle-weather.csv")

    # Generate the plot
    plt.figure(figsize=(8, 6))
    data.plot("precipitation", "temp_max", style='o')
    plt.xlabel("Precipitation")
    plt.ylabel("Temp Max")
    plt.title("Scatter Plot: Precipitation vs. Temp Max")
    
    # Calculate Pearson correlation
    pearson_corr = data["precipitation"].corr(data["temp_max"])

    # Perform T Test and get P value
    t_stat, p_value = stats.ttest_ind(data["precipitation"], data["temp_max"])

    # Convert the plot to base64 for embedding in HTML
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plot_url = f'data:Precipitation/png;base64,{plot_data}'

    # Render the template with plot URL and correlation values
    return render(request, 'Precipitation.html', {'plot_image_path': plot_url, 'pearson_corr': pearson_corr, 't_stat': t_stat, 'p_value': p_value})

def wind(request):
    # Read the data (assuming it's defined somewhere in your code)
    data = pd.read_csv("C:\\Users\\12082\\Downloads\\seattle-weather.csv")

    # Generate the plot
    plt.figure(figsize=(8, 6))
    data.plot("wind", "temp_max", style='o')
    plt.xlabel("wind")
    plt.ylabel("Temp Max")
    plt.title("Scatter Plot: Wind vs. Temp Max")
    
    # Calculate Pearson correlation
    pearson_corr = data["wind"].corr(data["temp_max"])

    # Perform T Test and get P value
    t_stat, p_value = stats.ttest_ind(data["wind"], data["temp_max"])

    # Convert the plot to base64 for embedding in HTML
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plot_url = f'data:wind/png;base64,{plot_data}'

    # Render the template with plot URL and correlation values
    return render(request, 'wind.html', {'plot_image_path': plot_url, 'pearson_corr': pearson_corr, 't_stat': t_stat, 'p_value': p_value})

def subplot(request):
    # Read the data (assuming it's defined somewhere in your code)
    data = pd.read_csv("C:\\Users\\12082\\Downloads\\seattle-weather.csv")

    # Create a figure and axis for the bar plot
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(1, 2, 2)

    # Create the bar plot using pandas
    data.drop(["date"], axis=1).plot(kind='bar', ax=ax, fontsize=12)

    # Customize the plot if needed (add titles, labels, etc.)
    plt.title("Bar Plot")
    plt.xlabel("X-Axis Label")
    plt.ylabel("Y-Axis Label")

    # Save the plot as a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Convert the plot to base64 for embedding in HTML
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plot_url = f'data:subplot/png;base64,{plot_data}'

    # Render the template with the plot URL
    return render(request, 'subplot.html', {'plot_image_path': plot_url})

def DataFrame(request):
    # Read the data (assuming it's defined somewhere in your code)
    data = pd.read_csv("C:\\Users\\12082\\Downloads\\seattle-weather.csv")

    # Drop the "date" column from the original DataFrame
    df = data.drop(["date"], axis=1)

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])

    # Calculate Q1 and Q3 for numeric columns
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)

    # Calculate IQR for numeric columns
    IQR = Q3 - Q1

    # Filter the DataFrame based on IQR for numeric columns
    filtered_df = df[~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Apply square root transformation to specific columns
    filtered_df['precipitation'] = filtered_df['precipitation'].apply(lambda x: x ** 0.5)
    filtered_df['wind'] = filtered_df['wind'].apply(lambda x: x ** 0.5)

    # Set Seaborn style
    sns.set(style="darkgrid")

    # Create subplots for histograms
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Create histograms with KDE plots
    sns.histplot(data=filtered_df, x="precipitation", kde=True, ax=axs[0, 0], color='green')
    sns.histplot(data=filtered_df, x="temp_max", kde=True, ax=axs[0, 1], color='red')
    sns.histplot(data=filtered_df, x="temp_min", kde=True, ax=axs[1, 0], color='skyblue')
    sns.histplot(data=filtered_df, x="wind", kde=True, ax=axs[1, 1], color='orange')

    # Save the plot as a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    # Convert the plot to base64 for embedding in HTML
    plot_data = base64.b64encode(buffer.getvalue()).decode()
    plot_url = f'data:DataFrame/png;base64,{plot_data}'

    # Render the template with the plot URL
    return render(request, 'DataFrame.html', {'plot_image_path': plot_url})

def accuracy(request):
    # Read the data
    data = pd.read_csv("C:\\Users\\12082\\Downloads\\seattle-weather.csv")

    # Drop the "date" column from the original DataFrame
    df = data.drop(["date"], axis=1)

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['number'])

    # Calculate Q1 and Q3 for numeric columns
    Q1 = numeric_df.quantile(0.25)
    Q3 = numeric_df.quantile(0.75)

    # Calculate IQR for numeric columns
    IQR = Q3 - Q1

    # Filter the DataFrame based on IQR for numeric columns
    filtered_df = df[~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Apply square root transformation to specific columns
    filtered_df['precipitation'] = filtered_df['precipitation'].apply(lambda x: x ** 0.5)
    filtered_df['wind'] = filtered_df['wind'].apply(lambda x: x ** 0.5)

    # Encode the "weather" column using LabelEncoder
    lc = LabelEncoder()
    df["weather"] = lc.fit_transform(df["weather"])

    # Prepare features (x) and target (y)
    x = filtered_df.drop(columns=["weather"]).values
    y = filtered_df["weather"].values

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)

    # Initialize and train KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    Knn = knn.score(x_test, y_test) * 100

    # Initialize and train Support Vector Machine (SVM) Classifier
    svm = SVC()
    svm.fit(x_train, y_train)
    Svm = svm.score(x_test, y_test) * 100

    # Initialize and train Gradient Boosting Classifier
    gbc = GradientBoostingClassifier(subsample=0.5, n_estimators=450, max_depth=5, max_leaf_nodes=25)
    gbc.fit(x_train, y_train)
    Gbc = gbc.score(x_test, y_test) * 100

    context = {'knn': Knn, 'svm': Svm, 'gbc': Gbc}

    return render(request, 'accuracy.html')

def prediction(request):
    if request.method == 'POST':
        form = WeatherPredictionForm(request.POST)
        if form.is_valid():
            # Get input data from the form
            precipitation = form.cleaned_data['precipitation']
            temp_max = form.cleaned_data['temp_max']
            temp_min = form.cleaned_data['temp_min']
            wind = form.cleaned_data['wind']
    # Read data from CSV file
    data = pd.read_csv("C:\\Users\\12082\\Downloads\\seattle-weather.csv")
    
    # Drop the "date" column from the original DataFrame
    df = data.drop(["date"], axis=1)
    
    # Convert numeric columns to numeric type, handling non-numeric values as NaN
    numeric_columns = ['precipitation', 'temp_max', 'temp_min', 'wind']
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Drop rows with NaN values in numeric columns
    df.dropna(subset=numeric_columns, inplace=True)
    
    # Apply transformations to specific columns
    df['precipitation'] = df['precipitation'].apply(lambda x: x ** 0.5)
    df['wind'] = df['wind'].apply(lambda x: x ** 0.5)
    
    # Encode string labels to numerical values
    label_encoder = LabelEncoder()
    df['weather'] = label_encoder.fit_transform(df['weather'])
    
    # Prepare features (x) and target (y)
    x = df.drop(columns=["weather"]).values
    y = df["weather"].values
    
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2)
    
    # Ignore warnings during model training
    warnings.filterwarnings('ignore')
    
    # Initialize and train XGBoost Classifier
    xgb = XGBClassifier()
    xgb.fit(x_train, y_train)
    
    # Calculate accuracy
    accuracy = xgb.score(x_test, y_test) * 100
    
    predicted_weather_label = None
    
    if request.method == 'POST':
        # Get input data from the form
        precipitation = float(request.POST['precipitation'])
        temp_max = float(request.POST['temp_max'])
        temp_min = float(request.POST['temp_min'])
        wind = float(request.POST['wind'])

        # Sample input data for prediction
        input_data = [[precipitation ** 0.5, temp_max, temp_min, wind ** 0.5]]
        
        # Predict weather code
        predicted_weather_code = xgb.predict(input_data)[0]
        
        # Reverse transform the numerical prediction back to the original string label
        predicted_weather_label = label_encoder.classes_[predicted_weather_code]

    context = {
        'accuracy': accuracy,
        'predicted_weather': predicted_weather_label
    }

    return render(request, 'prediction.html', context)

def homepage(request):
    return render(request,'homepage.html')

def statistics(request):
    return render(request,'statistics.html')