from markupsafe import Markup
import pandas as pd
from utils.fertilizer import fertilizer_dic
from werkzeug.utils import secure_filename
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from flask_pymongo import PyMongo
from flask import Flask, render_template, request, redirect, url_for, flash, session,    send_from_directory, jsonify    
from flask_pymongo import PyMongo
from datetime import datetime
import torch
import io
import pytesseract
from utils.disease import disease_dic
from torchvision import transforms
import os
import re
import cv2
from werkzeug.utils import secure_filename
from PIL import Image
from utils.model import ResNet9


app = Flask(__name__)


disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction


@ app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('homie.html', title = title)

# @ app.route('/homie')
# def homeieee():
#     title = 'Harvestify - Home'
#     return render_template('homie.html', title = title)



@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)



@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(float(request.form['nitrogen']))
    P = int(float(request.form['phosphorous']))
    K = int(float(request.form['pottasium']))
    # ph = float(request.form['ph'])

    df = pd.read_csv('Data/fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]

    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)

# Configure the MongoDB URI
app.config['MONGO_URI'] = 'mongodb://localhost:27017/community'
mongo = PyMongo(app)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define database collections
users = mongo.db.users
posts = mongo.db.posts
comments = mongo.db.comments
likes_dislikes = mongo.db.likes_dislikes

app.secret_key = 'your_secret_key'



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username is already in use
        existing_user = users.find_one({'username': username})
        if existing_user:
            flash('Username already exists. Choose a different one.')
        else:
            # Insert the new user into the database
            new_user = {'username': username, 'password': password}
            users.insert_one(new_user)
            flash('Registration successful. You can now log in.')
            return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users.find_one({'username': username, 'password': password})
        if user:
            session['user'] = username
            flash('Login successful.')
            return redirect(url_for('home_community'))
        else:
            flash('Invalid username or password.')

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home_community'))


@app.route('/home_community')
def home_community():
    if 'user' in session:
        user = session['user']
        all_posts = posts.find().sort('_id', -1)
        return render_template('home.html', user=user, posts=all_posts)
    return redirect(url_for('login'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/create_post', methods=['GET', 'POST'])
def create_post():
    if 'user' in session:
        if request.method == 'POST':
            post_content = request.form['content']
            post_date = datetime.now()
            
            # Handle image upload
            # Handle image upload in the 'create_post' route
            if 'image' in request.files:
                image = request.files['image']
                if image.filename != '':
                    image_filename = secure_filename(image.filename)
                    image.save(os.path.join(app.config['UPLOAD_FOLDER'], image_filename))
                    image_path = image_filename
                else:
                    image_path = None
            else:
                image_path = None


            new_post = {
                'user': session['user'],
                'content': post_content,
                'date': post_date,
                'image_path': image_path  # Store the image file name in MongoDB
            }

            # Insert the new post into the database
            posts.insert_one(new_post)
            flash('Post created successfully.')
            return redirect(url_for('home_community'))
        return render_template('create_post.html')
    return redirect(url_for('login'))


@app.route('/like_post/<post_id>')
def like_post(post_id):
    if 'user' in session:
        user = session['user']
        # Check if the user has already liked or disliked the post
        existing_like_dislike = likes_dislikes.find_one({'post_id': post_id, 'user': user})
        if existing_like_dislike:
            # User has already liked or disliked the post
            flash('You have already liked or disliked this post.')
        else:
            # Add a new like record
            likes_dislikes.insert_one({'post_id': post_id, 'user': user, 'type': 'like'})
            flash('You liked the post!')
    return redirect(url_for('home_community'))

@app.route('/dislike_post/<post_id>')
def dislike_post(post_id):
    if 'user' in session:
        user = session['user']
        # Check if the user has already liked or disliked the post
        existing_like_dislike = likes_dislikes.find_one({'post_id': post_id, 'user': user})
        if existing_like_dislike:
            # User has already liked or disliked the post
            flash('You have already liked or disliked this post.')
        else:
            # Add a new dislike record
            likes_dislikes.insert_one({'post_id': post_id, 'user': user, 'type': 'dislike'})
            flash('You disliked the post!')
    return redirect(url_for('home_community'))

@app.route('/comment_post/<post_id>', methods=['POST'])
def comment_post(post_id):
    if 'user' in session:
        user = session['user']
        comment_content = request.form['comment_content']
        # Add the comment to the comments collection along with the post_id and user information
        comments.insert_one({'post_id': post_id, 'user': user, 'content': comment_content})
        flash('Your comment was added.')
    return redirect(url_for('home_community'))

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  


UPLOAD_FOLDER = 'uploads_extract'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to enhance text
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Save the preprocessed image for reference
    preprocessed_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'preprocessed_image.png')
    cv2.imwrite(preprocessed_image_path, thresh)
    
    return preprocessed_image_path

@app.route('/text extract')
def index():
    return render_template('uploadimage.html')

@app.route('/extracted', methods=['POST'])
def extract_name_from_aadhar():
    # Ensure the request contains an image file
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image = request.files['image']
    
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Save the uploaded image to the upload folder
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image.filename))
        image.save(image_path)
        
        # Preprocess the image to enhance OCR accuracy
        preprocessed_image_path = preprocess_image(image_path)
        
        # Perform OCR on the preprocessed image
        text = pytesseract.image_to_string(Image.open(preprocessed_image_path))
        lines = text.split('\n')

        # Remove any empty lines from the list
        lines = [line.strip() for line in lines if line.strip()]

        # Print the list of lines
        for line in lines:
            print(line)
        
        pattern = r'\d+\.\d+'

        matches = re.findall(pattern, text)

        extracted_values = [float(match) for match in matches]

      
        
        if extracted_values:
            print(extracted_values)
            N = extracted_values[7]
            P = extracted_values[8]
            K = extracted_values[9]
            return render_template('extracted_details.html',N = N,K = K, P= P)
           
        else:
            return jsonify({'error': 'Na ho payega'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500    
 


if __name__ == '__main__':
    app.run(debug=True)
    