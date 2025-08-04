from django import forms

class ImageForm(forms.Form):
    image = forms.ImageField(required=True)
    message = forms.CharField(max_length=500)