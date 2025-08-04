from django.shortcuts import render
from .forms import ImageForm
import base64
from .utils import image_text
import json

# Create your views here.
def homepage(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = form.cleaned_data['image']
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            mime_type = image_file.content_type
            data = image_text.make_desc(image_data)
            print("LLM data:",data)
            response_str = data.get("response", "").strip()
            try:
                response_dict = json.loads(response_str)
            except json.JSONDecodeError:
                import ast
                response_dict = ast.literal_eval(response_str)
            print("Parsed response:", response_dict)
            desc = {
                'Name': response_dict.get('Name', 'Missing'),
                'Type_of_shop': response_dict.get('Type of shop', 'Missing'),
                'Pharmacy': response_dict.get('Pharmacy', 'Missing'),
                'Description': response_dict.get('Description', 'Missing'),
                'Address': response_dict.get('Address', 'Missing'),
                'Contact': response_dict.get('Contact', 'Missing'),
                'Other': response_dict.get('Other', 'Missing'),
            }
            return render(request, 'post.html', {'image_data': image_data,'mime_type': mime_type,'desc': desc})
    else:
        form = ImageForm()
    return render(request, 'get.html', {'form': form})