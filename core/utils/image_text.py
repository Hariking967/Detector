def extract_text_from_image(img):
    import cv2 as cv
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(img)
def image_to_description(img):
    from llama_cpp import Llama
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models\mistral-7b-instruct-v0.1.Q4_K_M.gguf")
    llm = Llama(
        model_path=model_path,
        chat_format="llama-2",
        n_ctx=2048
    )
    # content = extract_text_from_image('SampleImages/pharmBboard.jpg')
    content = extract_text_from_image(img)
    print(content)
    response = llm.create_chat_completion(
        messages=[
            {'role':"system", "content":"You are a helpful assistant that takes in a gibberish text which was extracted from a bill board image but as it is noisy you have to construct a proper description for that noisy unformatted text provided to you. Especially extract the name, type and a short descrioption of the type of the shop, address and contact info if provided. Few Notes make the response professional just in this format *Name: --, Type of shop: --, Description: --, Adress:--, Contact:--,Other: --*. Important: return in python dictionary format, if some attribute is missing then enter 'missing' in the dictionary.Strict Rule: Only return the python dictionary, use Type_of_shop and not Type of shop do not return \n characters at all return only the perfect python dictrionary only."},
            {'role':'user', 'content':content}
        ]
    )

    return {'response':response['choices'][0]['message']['content']}

def make_desc(image_data):
    import cv2 as cv
    import base64
    import numpy as np
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    return image_to_description(img)