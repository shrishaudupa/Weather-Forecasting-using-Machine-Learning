from django import forms

class WeatherPredictionForm(forms.Form):
    precipitation = forms.FloatField(label='Precipitation')
    temp_max = forms.FloatField(label='Maximum Temperature')
    temp_min = forms.FloatField(label='Minimum Temperature')
    wind = forms.FloatField(label='Wind Speed')
