# -*- coding: utf-8 -*-
from django import forms

class CompanyForm(forms.Form):
    """公司FOrm，包括公司名, 公司ID, 系统ID和关键词"""
    Name = forms.CharField(max_length=32)
    CompanyID = forms.CharField(max_length=64)
    SystemID = forms.CharField(max_length=64)
    Keywords = forms.CharField(max_length=1000)

class DeleteForm(forms.Form):
    """只能通过name来删除"""
    Name = forms.CharField(max_length=32)