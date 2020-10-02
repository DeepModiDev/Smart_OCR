
from django.db import models


class History(models.Model):
    uploaded_at = models.DateTimeField(auto_now_add=True)
    image_path = models.ImageField(upload_to='documents/')
    predicted_text = models.CharField(max_length=500, null=True)

    def __str__(self):
        return self.image_path
