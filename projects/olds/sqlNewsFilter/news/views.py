from django.shortcuts import render

# Create your views here.
from news import models
from django.shortcuts import HttpResponse, render_to_response


from news.forms import CompanyForm, DeleteForm


def db_insert(request):
    """add a record into DataBase"""
    if request.method == "POST":
        form = CompanyForm(request.POST)
        if form.is_valid():
            d = {"Name": request.POST['Name'],
                 "CompanyID": request.POST['CompanyID'],
                 "SystemID": request.POST['SystemID'],
                 "Keywords": request.POST['Keywords']}
            try:
                models.CompanyInfo.objects.create(**d)
            except:
                return HttpResponse("错误！项目名称重复！")
            else:
                return db_show(request)
    else:
        form = CompanyForm()
    return render_to_response("insert.html", {"form": form})

def db_delete(request):
    """delete a record from DataBase"""
    if request.method == "POST":
        form = DeleteForm(request.POST)
        if form.is_valid():
            name = request.POST['Name']
            models.CompanyInfo.objects.filter(Name=name).delete()
            return db_show(request)
    else:
        form = DeleteForm()
    return render_to_response("insert.html", {"form": form})


def db_show(request):
    """show the data in the database"""
    company_list = models.CompanyInfo.objects.all()
    return render(request, 'show.html', {'li': company_list})

def db_update(request):
    """update data in the database"""
    if request.method == "POST":
        form = CompanyForm(request.POST)
        if form.is_valid():
            name = request.POST['Name']
            d = {"CompanyID": request.POST['CompanyID'],
                 "SystemID": request.POST['SystemID'],
                 "Keywords": request.POST['Keywords']}
            models.CompanyInfo.objects.filter(Name=name).update(**d)
            return db_show(request)
    else:
        form = CompanyForm()
    return render_to_response("update.html", {"form": form})