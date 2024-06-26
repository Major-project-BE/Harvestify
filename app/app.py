import json
from markupsafe import Markup
import pandas as pd
from utils.fertilizer import fertilizer_dic
from werkzeug.utils import secure_filename
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from flask_pymongo import PyMongo
from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory, jsonify    
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
from bson import ObjectId
import datetime
from twilio.rest import Client


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
    crop_list = {'rice':'https://media.istockphoto.com/id/497711693/photo/rice-field.webp?b=1&s=170667a&w=0&k=20&c=ziH0h9qotY_MkbWl7f_M1mUZ12ghBzNLselY8OT8D84=',
                 'maize':'https://img.freepik.com/free-photo/beautiful-shot-cornfield-with-blue-sky_181624-20783.jpg',
                 'chickpea':'https://t4.ftcdn.net/jpg/02/77/58/17/360_F_277581792_trTRdyvnE9H5rPLt1WDLUyHK7ZJ8FAny.jpg',
                 'kidneybeans':'https://www.crops.org/files/images/news/20160823-143936-hand-holding-kidney-beans-800x600.jpg',
                 'pigeonpeas':'https://static.vecteezy.com/system/resources/previews/006/540/162/non_2x/green-pigeon-pea-field-in-india-photo.JPG',
                 'mothbeans':'https://www.feedipedia.org/sites/default/files/images/Vigna-aconitifolia_leaves%26flowers-MJussoorie%20Chakrata%20road%20near%20Bharatkhai-1-DSC09876.jpg',
                 'mungbean':'https://www.epicgardening.com/wp-content/uploads/2021/10/Mung-bean-plant.jpg',
                 'blackgram':'https://www.agrifarming.in/wp-content/uploads/2015/04/Black-Gram-Production..jpg',
                 'lentil':'https://media.sciencephoto.com/e7/70/09/17/e7700917-800px-wm.jpg',
                 'pomegranate':'https://agrosiaa.com/uploads/userdata/blogs/24/16143999001.png',
                 'banana':'https://media.istockphoto.com/id/471467855/photo/banana-tree.jpg?s=612x612&w=0&k=20&c=gPOgedQhaMK26gOdUEcnnDiKGOnBGNGc199ajc0TGYo=',
                 'mango':'https://media.istockphoto.com/id/601122142/photo/crop-of-sun-kissed-mango-fruit-ripening-on-tree.jpg?s=612x612&w=0&k=20&c=LWqDqwt6SV5ye5WQs8M3xUmqiQNgLxu41HWxj4LvEEs=',
                 'grapes':'https://www.tpci.in/indiabusinesstrade/wp-content/uploads/2020/12/Grape-export.jpg',
                 'watermelon':'https://horticulture.punjab.gov.in/images/crops/Watermelon1.jpg',
                 'muskmelon':'https://www.agrifarming.in/wp-content/uploads/2015/05/Growing-Cantaloupe..jpg',
                 'apple':'https://www.thestatesman.com/wp-content/uploads/2022/09/The-Apple-cultivation-story-is-full-of-challenges-in-India-1.jpg',
                 'orange':'https://www.apnikheti.com/upload/crops/4257idea99oranges-on-citrus-tree.jpg',
                 'papaya':'https://blog.agribegri.com/public/blog_images/papaya-farming-guide-600x400.jpg',
                 'coconut':'https://www.asiafarming.com/wp-content/uploads/2015/09/Coconut-Cultivation1.png',
                 'cotton':'https://bloximages.chicago2.vip.townnews.com/pinalcentral.com/content/tncms/assets/v3/editorial/9/63/963d234e-9fc7-5042-be5d-11f691fb8105/5c093cbb3514c.image.jpg',
                 'jute':'https://t3.ftcdn.net/jpg/05/61/99/80/360_F_561998023_YmOc0Qe3VTK0o5uhJ9eH3BSX49z5dDVl.jpg',
                 'coffee':'https://media.istockphoto.com/id/1321031195/photo/coffee-ready-for-harvest.jpg?s=612x612&w=0&k=20&c=Oy05CZ7VZ9HtsjtE9ttFnzkRg0kEEelyHTCUondHQ7w='}
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
    crop_image = crop_list[crop_name]
    return render_template('fertilizer-result.html', recommendation=response,crop_image=crop_image,crop_name=crop_name.capitalize(), title=title)

@app.route('/disease-predict', methods=['GET', 'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease_new.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)
           
            prediction = Markup(str(disease_dic[prediction]))

            return render_template('disease_new-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease_new.html', title=title)

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



@app.route('/register', methods=['POST','GET'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username is already in use
        existing_user = users.find_one({'username': username})
        if existing_user:
            flash('Username already exists. Choose a different one.','error')
        else:
            # Insert the new user into the database
            new_user = {'username': username, 'password': password}
            users.insert_one(new_user)
            flash('Registration successful. You can now log in.','success')
            return redirect(url_for('login'))

    return render_template('login_new.html')

@app.route('/login', methods=['POST','GET'])
def login():
    title = "Harvestify - Community"
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users.find_one({'username': username, 'password': password})
        if not user:
            flash('Invalid Username or Password.','error')
        else:
            session['user'] = username
            return redirect(url_for('home_community'))
            
            

    return render_template('login_new.html',title=title)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home_community'))




@app.route('/home_community')
def home_community():
    
    if 'user' in session:
        user = session['user']
        all_posts = posts.find().sort('_id', -1)
        all_comments = comments.find().sort('_id',-1)
        convert = list(all_comments)
        
        for i in convert:
            i['post_id'] = ObjectId(i['post_id'])

        return render_template('home.html', user=user, posts=all_posts,comments= convert)
    return redirect(url_for('login'))



# @app.route('/policy')
# def policcy():
    
#     return render_template('policy.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)



@app.route('/create_post', methods=['GET', 'POST'])
def create_post():
    if 'user' in session:
        if request.method == 'POST':
            post_content = request.form['content']
            post_date_extracted = datetime.datetime.now()
            post_date = post_date_extracted.strftime("%c")
            no_likes = 0
            no_dislikes = 0
            
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
                'image_path': image_path , # Store the image file name in MongoDB
                'no_likes' : no_likes,
                'no_dislikes' : no_dislikes
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
            posts.update_one({'_id': ObjectId(post_id)}, {'$inc': {'no_likes': 1}})
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
            posts.update_one({'_id': ObjectId(post_id)}, {'$inc': {'no_dislikes': 1}})
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
    


@app.route('/policy')
def policcy():
    # Read the JSON data from the file
    with open('amazon_data.json', 'r') as f:
        data = json.load(f)
    return render_template('policy.html', data=data)



# Connecting Policy DB to Flask
app.config['MONGO_URI'] = 'mongodb://localhost:27017/Policies'
mongo = PyMongo(app)

UPLOAD_FOLDER1 = 'upload_aadhar'
app.config['UPLOAD_FOLDER1'] = UPLOAD_FOLDER1
if not os.path.exists(UPLOAD_FOLDER1):
    os.makedirs(UPLOAD_FOLDER1)

UPLOAD_FOLDER2 = 'upload_income_cert'
app.config['UPLOAD_FOLDER2'] = UPLOAD_FOLDER2
if not os.path.exists(UPLOAD_FOLDER2):
    os.makedirs(UPLOAD_FOLDER2)


# Defining new collection for policies
policy_data = mongo.db.policy_data


# Twilio configuration
TWILIO_ACCOUNT_SID = 'AC03c8248bd59994d4985d7f8930783cb5'
TWILIO_AUTH_TOKEN = '779d6c2664307f6a9177a9f4c8b2dd16'
TWILIO_PHONE_NUMBER = '+17143123626'
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@app.route('/upload_aadhar/<filename>')
def uploaded_file1(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER1'], filename)

@app.route('/upload_income_cert/<filename>')
def uploaded_file2(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER2'], filename)

@app.route('/apply_policy/<policy_name>',methods=['POST','GET'])
def apply_policy(policy_name):
    if 'user' in session:
        if request.method == 'POST':
            policy_name = request.form['policy_name']
            name = request.form['name']
            address = request.form['address']
            age = request.form['age']
            gender = request.form['gender']

            
            aadhar_copy = request.files['aadhar']
            if aadhar_copy.filename != '':
                aadhar_filename = secure_filename(aadhar_copy.filename)
                aadhar_copy.save(os.path.join(app.config['UPLOAD_FOLDER1'], aadhar_filename))
                aadhar_filename = aadhar_filename
            else:
                aadhar_filename = None
            

            
            income_copy = request.files['incomeCertificate']
            if income_copy.filename != '':
                income_filename = secure_filename(income_copy.filename)
                income_copy.save(os.path.join(app.config['UPLOAD_FOLDER2'], income_filename))
                income_filename = income_filename
            else:
                income_filename = None
            
           

            phone_no = '+91'+str(request.form['phone_no'])

            policy_doc ={
                'name' : name,
                'age' : age,
                'gender' : gender,
                'address' : address,
                'phone_no': phone_no,
                'policy_name': policy_name,
                'aadhar_filename' : aadhar_filename,
                'income_filename' : income_filename
            }

            # Send message using Twilio
            message = client.messages.create(
                body=f"Your form for ({policy_name}) has been submitted successfully. We will contact you in future for the further process, ({policy_name}) के लिए आपका फॉर्म सफलतापूर्वक सबमिट कर दिया गया है। आगे की प्रक्रिया के लिए हम भविष्य में आपसे संपर्क करेंगे। आपकी माँ का भोसड़ा",
                from_=TWILIO_PHONE_NUMBER,
                to= phone_no
            )

            policy_data.insert_one(policy_doc)
            flash('You have Registered for the Policy')
            return redirect(url_for('policcy'))
       
        return render_template('register_policy.html',policy_name=policy_name)
    return redirect(url_for('login'))




pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  


UPLOAD_FOLDERR = 'uploads_extract'
app.config['UPLOAD_FOLDERR'] = UPLOAD_FOLDERR
if not os.path.exists(UPLOAD_FOLDERR):
    os.makedirs(UPLOAD_FOLDERR)

def preprocess_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to enhance text
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Save the preprocessed image for reference
    preprocessed_image_path = os.path.join(app.config['UPLOAD_FOLDERR'], 'preprocessed_image.png')
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
        image_path = os.path.join(app.config['UPLOAD_FOLDERR'], secure_filename(image.filename))
        image.save(image_path)
        
        # Preprocess the image to enhance OCR accuracy
        preprocessed_image_path = preprocess_image(image_path)
        
        # Perform OCR on the preprocessed image
        text = pytesseract.image_to_string(Image.open(preprocessed_image_path))
        lines = text.split('\n')

        # Remove any empty lines from the list
        lines = [line.strip() for line in lines if line.strip()]

        ##
        my_list = []
        for line in lines:
            if "Nitrogen" in line:
                my_list.append(line)
            if "Phosphorous" in line:
                my_list.append(line)
            if "Potassium" in line:
                my_list.append(line)
        my_text = "\n".join(my_list)
        print(my_text)

        
        pattern = r'\d+\.\d+'

        matches = re.findall(pattern, my_text)
        print(matches)

        extracted_values = [float(match) for match in matches]
        
        if extracted_values:
            print(extracted_values)
            N = extracted_values[0]
            P = extracted_values[1]
            K = extracted_values[2]
            return render_template('extracted_details.html',N = N,K = K, P= P)
           
        else:
            return jsonify({'error': 'Na ho payega'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500    
 


if __name__ == '__main__':
    app.run(debug=True)
    