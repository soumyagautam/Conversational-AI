import google.generativeai as palm
from flask import Flask, render_template, request
from transformers import BartTokenizer, BartForConditionalGeneration
from googlesearch import search

API_KEY = "YOUR API KEY HERE"

palm.configure(api_key=API_KEY)
models = [str(model.name) for model in palm.list_models()]
model_id = models[1]

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template("index.html", google="")


@app.route("/submit", methods=["POST"])
def submit():
    prompt = request.form["text"]

    if prompt == "" or prompt.isspace() is True:
        return render_template("index.html", output="Please give some input")

    completion = palm.generate_text(
        model=model_id,
        prompt=prompt,
        temperature=0.99,
        max_output_tokens=800,
    )

    result = completion.result

    google_results = []
    for j in search(prompt):
        google_results.append(j)

    input_ids = tokenizer.encode(result, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids, max_length=500, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return render_template("index.html", output=result, ques=prompt, summary=summary, google=google_results)


if __name__ == "__main__":
    app.run(debug=True)
