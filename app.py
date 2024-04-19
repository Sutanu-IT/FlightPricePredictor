from flask import Flask, request, render_template, jsonify
# Alternatively can use Django, FastAPI, or anything similar
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application

# @app.route('/')
# def home_page():
#     return render_template('index.html')

@app.route('/', methods = ['POST', "GET"])
def predict_datapoint(): 
    if request.method == "GET": 
        return render_template("form.html")
    else: 
        data = CustomData(
            days_left = float(request.form.get('days_left')),
            airline = request.form.get('airline'),
    		source_city = request.form.get('source_city'),
    		departure_time = request.form.get('departure_time'),
    		stops = request.form.get('stops'),
    		arrival_time = request.form.get('arrival_time'),
    		destination_city = request.form.get('destination_city'),
    		class_type = request.form.get('class') 
        )
    new_data = data.get_data_as_dataframe()
    predict_pipeline = PredictPipeline()
    pred = predict_pipeline.predict(new_data)

    results = round(pred[0],2)

    return render_template("form.html", final_result = results)

if __name__ == "__main__": 
    app.run(host = "0.0.0.0", debug= True)

#http://127.0.0.1:5000/ in browser