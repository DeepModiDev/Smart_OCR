from django.db import models

from .models import History

class HistoryForm(forms.ModelForm):
    class Meta:
        model = History
        fields = ('uploaded_at', 'image_path', 'predicted_text')