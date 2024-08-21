from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the XGBoost model
model = joblib.load('xgb_model3 (1).pkl')

# Sample data structure for food items
food_items = [
    {'name': 'Pizza', 'original_price': 100.00, 'discount': 2.00, 'discounted_price': 98.00, 'rating': 4.0, 'image': 'pizza.jpeg'},
    # {'name': 'Burger', 'original_price': 5.00, 'discount': 1.00, 'discounted_price': 4.00, 'rating': 3.5, 'image': 'burger.jpeg'},
    # Add more items as needed
]

# Homepage route
@app.route('/')
def index():
    return render_template('index.html', food_items=food_items)

# Admin page route
@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        name = request.form['name']
        original_price = float(request.form['original_price'])
        discount = float(request.form['discount'])
        discounted_price = original_price - discount
        rating = float(request.form['rating'])  # New line to handle rating
        image = request.files['image']

        image_filename = os.path.join('static/images', image.filename)
        image.save(image_filename)

        food_items.append({
            'name': name,
            'original_price': original_price,
            'discount': discount,
            'discounted_price': discounted_price,
            'rating': rating,  # Store the rating
            'image': image.filename
        })

        return redirect(url_for('index'))
    
    return render_template('admin.html')

# Prediction page route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = {
            'original price': [float(request.form['original_price'])],
            'sales_volume': [float(request.form['sales_volume'])],
            'inventory_level': [float(request.form['inventory_level'])],
            'discount_last_week': [float(request.form['discount_last_week'])],
            'day_of_week': [int(request.form['day_of_week'])],
            'month': [int(request.form['month'])],
            'lag_sales': [float(request.form['lag_sales'])]
        }
        
        df = pd.DataFrame(data)
        predicted_optimal_price = model.predict(df)[0]

        return render_template('prediction.html', predicted_price=predicted_optimal_price)
    
    return render_template('prediction.html', predicted_price=None)

if __name__ == '__main__':
    app.run(debug=True)
