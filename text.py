import io
import os

def detect_text(path):
    """Detects text in the file."""
    from google.cloud import vision
    import io
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\pys1\\Downloads\\emerald-rhythm-332904-7dd050962f9f.json'
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    text_list = list(map(lambda x: x.description, texts))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return text_list

print(detect_text('text_files/wakeupcat.jpg'))

