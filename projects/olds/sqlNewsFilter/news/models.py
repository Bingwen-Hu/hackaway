from django.db import models

# Create your models here.
class CompanyInfo(models.Model):
    Name = models.CharField(max_length=32, unique=True)
    CompanyID = models.CharField(max_length=64)
    SystemID = models.CharField(max_length=64)
    Keywords = models.TextField()