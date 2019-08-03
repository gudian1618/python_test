from django.contrib import admin

# Register your models here.
from blog.models import BlogArticles

admin.site.register(BlogArticles)