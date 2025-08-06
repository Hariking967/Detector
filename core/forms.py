from django import forms

class ImageForm(forms.Form):
    image = forms.ImageField(required=True)
    message = forms.CharField(widget=forms.Textarea(
        attrs={
            'row': 5,
            'cols': 40,
            'placeholder': "Enter your prompt here..."
        }
    ))