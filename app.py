from flask import Flask, render_template, request, flash, session, redirect, url_for
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import logging
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static', 'Uploads')
app.secret_key = 'your_secret_key'

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Forgery Detection Class
class DetectForgery:
    def __init__(self, image, response_thresh=0.003):
        self.image = image
        self.response_thresh = response_thresh
        self.keypoints = None
        self.descriptors = None

    def sift_features(self):
        sift = cv2.SIFT_create()
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        kps, descs = sift.detectAndCompute(gray, None)

        # Filter by keypoint response
        filtered_kps, filtered_descs = [], []
        for kp, desc in zip(kps, descs):
            if kp.response >= self.response_thresh:
                filtered_kps.append(kp)
                filtered_descs.append(desc)

        self.keypoints = filtered_kps
        self.descriptors = np.array(filtered_descs) if filtered_descs else None
        return self.keypoints, self.descriptors

    def detect(self, eps=44, min_samples=2, use_pca=True, n_pca_components=30):
        if self.descriptors is None or len(self.descriptors) < min_samples:
            return None, 0  # Not enough descriptors

        # Apply PCA
        reduced_desc = self.descriptors
        if use_pca and len(self.descriptors) >= n_pca_components:
            pca = PCA(n_components=n_pca_components)
            reduced_desc = pca.fit_transform(self.descriptors)

        # DBSCAN Clustering
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(reduced_desc)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters == 0:
            return None, 0  # Likely original

        # Collect cluster points
        cluster_pts = [[] for _ in range(n_clusters)]
        for idx, label in enumerate(labels):
            if label != -1:
                pt = (int(self.keypoints[idx].pt[0]), int(self.keypoints[idx].pt[1]))
                cluster_pts[label].append(pt)

        forgery = self.image.copy()
        clusters_used = 0

        for pts in cluster_pts:
            if len(pts) > 1:
                xs, ys = zip(*pts)
                dx, dy = max(xs) - min(xs), max(ys) - min(ys)
                if dx < 50 and dy < 50:
                    continue  # Skip tight clusters

                for i in range(1, len(pts)):
                    cv2.line(forgery, pts[0], pts[i], (0, 0, 255), 2)
                clusters_used += 1

        if clusters_used > 0:
            return forgery, 1  # Tampered
        else:
            return None, 0  # Original

# Authentication Check Decorator
def login_required(f):
    def wrap(*args, **kwargs):
        if not session.get('logged_in'):
            flash('Please log in first.', 'error')
            return render_template('index.html', show_login_modal=True, logged_in=False)
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if email == "abcd@gmail.com" and password == "1234":
            session['logged_in'] = True
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'error')

    return render_template('index.html', show_login_modal=True, logged_in=False)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def home():
    forgery_image = None
    original_image = None
    detection_result = None

    if request.method == 'POST':
        logging.debug(f"request.files: {request.files}")
        logging.debug(f"request.form: {request.form}")

        if 'image' not in request.files:
            logging.debug("No file part in the request")
            flash('No file part in the request. Please select a file.', 'error')
            return render_template('index.html', logged_in=True, show_login_modal=False)

        file = request.files['image']
        if file.filename == '':
            logging.debug("No file selected")
            flash('No file selected. Please choose a file to upload.', 'error')
            return render_template('index.html', logged_in=True, show_login_modal=False)

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logging.debug(f"File saved to: {file_path}")

        image = cv2.imread(file_path)
        if image is None:
            logging.debug("Failed to load image")
            flash('Failed to load the image. Please try a different file.', 'error')
            return render_template('index.html', logged_in=True, show_login_modal=False)

        # Process the image for forgery detection
        detector = DetectForgery(image, response_thresh=0.003)
        detector.sift_features()
        forgery, pred = detector.detect(eps=44, min_samples=2)

        if forgery is not None and pred == 1:
            forgery_filename = 'forgery_' + filename
            forgery_path = os.path.join(app.config['UPLOAD_FOLDER'], forgery_filename)
            cv2.imwrite(forgery_path, forgery)
            forgery_image = forgery_filename
            original_image = filename
            detection_result = "Tampered Region Detected"
        else:
            original_image = filename
            detection_result = "No Tampering Detected"
            flash('No tampering detected in the image.', 'info')

    return render_template('index.html', original=original_image, forgery=forgery_image, 
                         detection=detection_result, logged_in=True, show_login_modal=False)

if __name__ == '__main__':
    app.run(debug=True)