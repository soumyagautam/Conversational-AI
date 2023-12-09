import google.generativeai as palm
from flask import Flask, render_template, request

API_KEY = "YOUR API KEY HERE"

palm.configure(api_key=API_KEY)
models = [str(model.name) for model in palm.list_models()]
model_id = models[1]

app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/submit", methods=["POST"])
def submit():
    prompt = request.form["text"]

    if prompt is "":
        return render_template("index.html", output="Please give some input")

    completion = palm.generate_text(
        model=model_id,
        prompt=prompt,
        temperature=0.99,
        max_output_tokens=800,
    )

    result = completion.result

    return render_template("index.html", output=result)


if __name__ == "__main__":
    app.run(debug=True)
