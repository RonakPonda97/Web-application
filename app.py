from flask import Flask, render_template, request, jsonify
import pickle
import sklearn

app = Flask(__name__,template_folder="View",static_folder="Resources",static_url_path="")

# Load the 'Random Forest' classifier
with open('random_forest_classifier.pkl', 'rb') as f:
    random_forest_classifier = pickle.load(f)

# Define a route to serve the index.html page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to serve the team_prediction.html page
@app.route('/team-prediction', methods=['GET', 'POST'])
def team_prediction():
    if request.method == 'POST':
        data = request.form  # You can access form data here
        # Process the data and perform prediction using the random_forest_classifier

        # For example, if you want to return a prediction result
        prediction = random_forest_classifier.predict(data)  # Adjust this based on your data format
        return render_template('team_prediction.html', prediction=prediction)

    return render_template('team_prediction.html')  # Render the page initially

#if __name__ == '__main__':
app.run()
