from markupsafe import Markup
import pandas as pd
from utils.fertilizer import fertilizer_dic
from werkzeug.utils import secure_filename
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from flask_pymongo import PyMongo
from flask import Flask, render_template, request, redirect, url_for, flash, session,    send_from_directory
from flask_pymongo import PyMongo
from datetime import datetime
import os



app = Flask(__name__)


@ app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index.html', title = title)


@ app.route('/fertilizer')
def fertilizer_recommendation():
    title = 'Harvestify - Fertilizer Suggestion'

    return render_template('fertilizer.html', title=title)



@ app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = 'Harvestify - Fertilizer Suggestion'

    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
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




if __name__ == '__main__':
    app.run(debug=False)
    