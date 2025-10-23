from django.urls import path
from .views import PredictView

urlpatterns = [
    # Esto le dice a Django:
    # "Cuando la URL principal te envíe tráfico, 
    # usa la URL 'predict/' para activar la PredictView"
    path('predict/', PredictView.as_view(), name='predict'),
]