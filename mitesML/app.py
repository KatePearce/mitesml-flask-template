from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            
            input_data = [float(x) for x in request.form.values()]
            input_data = np.array(input_data).reshape(1, -1)
            
            # makes predictions
            
            prediction_num = model.predict(input_data)[0]
            prediction = "Cancerous"
            if prediction_num == 1:
                prediction = "Not Cancerous"
                
            print(prediction)

            return render_template('index.html', prediction=prediction)
        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)