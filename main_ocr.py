import requests
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
import numpy as  np 
import torch
import cv2 
import time
import pandas as pd
import os 
import matplotlib.image as mpimg

# Load the YOLOv5 object detection model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#Get the years of garfield comics 
years =  list(range(1978, 2024))
months = list(range(1,13))

# Define the URL of the Garfield comics website
url = 'http://pt.jikos.cz/garfield/2011/9/'

# Send a request to the website and get the HTML content
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all the comic images on the page
comic_images = soup.find_all('img')

# Initialize an empty dictionary to store the OCR data
data = {'image_name': [], 'ocr_text': []}


# Loop through the comic images and extract the OCR text
for index, image in enumerate(comic_images):
    
    image_url = image['src']
    if 'http' in image_url: 
        # Retry failed requests a few times
        for retry in range(3):
            try:
                response = requests.get(image_url, timeout=10)
                break
            except requests.exceptions.RequestException:
                print(f'Retrying request {retry + 1} for image {image_url}')
                time.sleep(1)

        # Save the image to disk
        with open(f'garfield{index}.jpg', 'wb') as f:
            f.write(response.content)


        #img = mpimg.imread(os.curdir + f'/garfield{index}.png')
        img = Image.open(f'garfield{index}.jpg')
        img.save(f'garfield{index}.png')
        img = cv2.imread(f'garfield{index}.png')
      
        # Create a binary mask to extract the text from the speech bubbles
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 20, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        img_masked = cv2.bitwise_and(img, img, mask=mask)

        # Convert the image to grayscale and apply thresholding
        img_gray = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
        img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


        # Perform OCR on the image using Tesseract
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        ocr_text = pytesseract.image_to_string(img_thresh, lang='eng')
        print(f'Garfield comic {index}')
        
        # Remove newlines and leading/trailing spaces
        ocr_text = ocr_text.replace('\n', ' ')
        ocr_text = ocr_text.strip()

        
        # Add the OCR data to the dictionary
        data['image_name'].append(f'garfield{index}.jpg')
        data['ocr_text'].append(ocr_text)
        
         # Delete the image file from disk
        cv2.imwrite(f'garfield{index}.jpg', img_thresh)
        cv2.imwrite(f'garfield{index}.png', img_thresh)
        # Add a delay between requests to avoid overloading the server
        time.sleep(1)

# Convert the dictionary to a pandas dataframe
df = pd.DataFrame(data)
print(df.head())

# Write the dataframe to a CSV file
df.to_csv('garfield_ocr.csv', index=False)