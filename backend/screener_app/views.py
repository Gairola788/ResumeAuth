from django.shortcuts import render
from django.http import HttpResponse
from .models import Resume

def home(request):
    return HttpResponse("ResumeAuth is live 🚀")

def about(request):
    return HttpResponse("This is the about page of ResumeAuth.")