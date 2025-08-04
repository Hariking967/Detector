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
            llm = image_text.init_llm()
            image_file = form.cleaned_data['image']
            message = form.cleaned_data['message']
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
            mime_type = image_file.content_type
            reply = image_text.make_desc(llm, image_data, message)
            print("LLM data:",reply)
            # response_str = data.get("response", "").strip()
            # try:
            #     response_dict = json.loads(response_str)
            # except json.JSONDecodeError:
            #     import ast
            #     response_dict = ast.literal_eval(response_str)
            desc = reply
            return render(request, 'post.html', {'image_data': image_data,'mime_type': mime_type,'desc': desc})
        else:
            print(form.errors)
    else:
        form = ImageForm()
    return render(request, 'get.html', {'form': form})