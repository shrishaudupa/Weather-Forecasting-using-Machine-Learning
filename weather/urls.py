"""
URL configuration for weather project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from forecast import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('table/',views.head),
    path('plot/',views.plot),
    path('percentage/',views.percentage_of_weather_condition),
    path('histogram/',views.generate_plots),
    path('violin/',views.violin),
    path('box/',views.boxplot),
    path('boxp/',views.boxplott),
    path('boxpw/',views.boxplotwind),
    path('minimum/',views.minimumtemp),
    path('corelation/',views.corelation),
    path('Precipitation/',views.Precipitation),
    path('wind/',views.wind),
    path('subplot/',views.subplot),
    path('DataFrame/',views.DataFrame),
    path('accuracy/',views.accuracy,name='accuracy'),
    path('prediction/',views.prediction,name='prediction'),
    path('statistics/',views.statistics,name='statistics'),
    path('homepage/',views.homepage,name='home'),

]
