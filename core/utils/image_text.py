def init_llm():
    from llama_cpp import Llama
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models\mistral-7b-instruct-v0.1.Q4_K_M.gguf")
    llm = Llama(
        model_path=model_path,
        chat_format="llama-2",
        n_ctx=2048
    )
    return llm

def extract_text_from_image(img):
    import cv2 as cv
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(img)
def image_to_description(llm, img, msg):
    # content = extract_text_from_image('SampleImages/pharmBboard.jpg')
    content = extract_text_from_image(img)
    print(content)
    response = llm.create_chat_completion(
        messages=[
            {'role':"system", "content":"You are a helpful assistant that takes in a gibberish text which was extracted from a bill board image but as it is noisy you have to construct a proper description for that noisy unformatted text provided to you. Especially extract the name, type and a short descrioption of the type of the shop, address and contact info if provided. Few Notes make the response professional just in this format *Name: --, Type of shop: --, Description: --, Adress:--, Contact:--,Other: --*. Important: Return as a string."},
            {'role':'user', 'content':content + f"Prompt message: {msg}"}
        ]
    )

    return {'response':response['choices'][0]['message']['content']}

def simple_llm(llm, *prompts):
    messages = [{"role":"system", "content":"You are an useful agent that accepts multiple user inputs and combine them in a reasonable logical order and return it.Note: If you find multiple detected objects group them like 'apple apple apple orange orange 'should be 'there are 3 apples and 2 oranges in the image provided'. Also try to find relation between the prompts sent by the user. IMPORTANT: After performing logical relation finding and object grouping if possible then return as a nice appropriate html markup format."}]
    for prompt in prompts:
        messages.append({"role":"user", "content":prompt})
    print("messages:",messages)
    response = llm.create_chat_completion(
        messages=messages
    )
    return response['choices'][0]['message']['content']
def yolo_desc(img):
    from ultralytics import YOLO
    model = YOLO('models/yolov8n.pt')
    results = model(img)
    labels = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = results[0].names[cls_id]
        labels.append(class_name)
    print(f"Detected Labels: " + ",".join(map(str,labels)))
    return f"Detected Labels: " + ",".join(map(str,labels))

def make_desc(llm, image_data, msg):
    import cv2 as cv
    import base64
    import numpy as np
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    tessseract_desc = image_to_description(llm, img, msg)
    print("tesseract_desc", tessseract_desc)
    yolo_labels = yolo_desc(img)
    if (yolo_labels.strip() == 'Detected Labels:'):
        reply = simple_llm(llm, tessseract_desc['response'])
    else:
        reply = simple_llm(llm, tessseract_desc['response'], yolo_labels)
    print("reply: ", reply)
    return reply