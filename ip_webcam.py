import requests
from io import BytesIO
from PIL import Image
import time
import os

# IP Webcam server url
camera_url = 'http://192.168.0.27:8080/shot.jpg'

while True:
    # Use urllib to get the image and convert into a cv2 usable format
    try:
        res = requests.get(camera_url)
        img = Image.open(BytesIO(res.content))
        img.save('shot.jpg')
        print('Image captured. Processing...')

        server_ip = '54.213.101.65'
        print('Saved. Uploading to ', server_ip)
        try:
            os.system("scp -i ~/.ssh/UbuntuServerPrivateKey.pem 'shot.jpg' \
                ubuntu@" + server_ip + ":/home/ubuntu/server/shot.jpg")
            print('Done.')
        except e:
            print('Error in uploading: ', e.message())

    except:
        print('Camera not found! Retrying...')

    #To give the processor some less stress
    time.sleep(0.1)
