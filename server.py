from flask import Flask, render_template, request, Markup, flash, Markup,make_response
import pandas
import sys
sys.path.append("./DS_insight")
sys.path.append("./data")
import article_key_extraction_ner as keyNER
from recommendation import *
from summarizer import Summarizer
import spacy # with pretrained model to recognize entities
from spacy import displacy
from pathlib import Path
nlp = spacy.load("en_core_web_lg")
# Create the application object
app = Flask(__name__)
df=pandas.read_csv('summary.csv')
#df=keyNER.preprocessing('./data/cleaned.csv')
@app.route("/")
def home_page():   
	return render_template('index.html')


@app.route('/entity', methods=['POST'])
def my_form_post():
	text = request.form['text']
	print(text)
	doc = nlp(text)
	print([(X.text, X.label_) for X in doc.ents])
	svg = displacy.render(doc, style='ent')
	svg = Markup(svg)
	flash(svg)

	return render_template('index.html')


@app.route('/entity', methods=['GET', 'POST'])
def my_form_get():
	text = request.args.get('ner')
	#print(type(t))

	#text = str(t.encode('utf-8'))
	doc = nlp(text)

	print([(X.text, X.label_) for X in doc.ents])
	svg = displacy.render(doc, style='ent', minify=True)

	svg = svg.replace(">b'", ">").replace("'</", "</")
	resp = make_response(svg)
	resp.mimetype = 'text/plain'
	return resp


@app.route('/output',methods=['get','post'])
def recommendation_output():
	#
	# Pull input
	some_input = request.args.get('user_input')

	# Case if empty
	if some_input == "":
		return render_template("index.html",
							   my_input=some_input,
							   my_form_result="Empty")
	else:
		#some_output="Double check your web!!Something wrong!!"
		#print(request.form)
		#if 'entity' in request.form :
			#some_output =keyNER.summNER(some_input)
			#doc = nlp(some_output)
			#some_output=doc.ents
			#print([(X.text, X.label_) for X in doc.ents])
			#svg = displacy.render(doc, style='ent')
			#svg = Markup(svg)
			#flash(svg)
		#elif 'recommend' in request.form:
		svg = displacy.render(doc_try, style='ent', page=True)
		output = Path('./sentence.svg')
		output.open("w", encoding="utf-8").write(svg)
		some_output = get_close_docs(some_input,df)

	#some_number = 3
	some_image = "panda.gif"
	return render_template("index.html",
						   my_input=some_input,
						   my_output0=some_output[1],
						   my_output1=some_output[2],
						   my_output2=some_output[3],
						   my_output3=some_output[4],
						   my_output4=some_output[5],
						   #my_number=some_number,
						   my_img_name=some_image,
						   my_form_result="NotEmpty")


# start the server with the 'run()' method
if __name__ == "__main__":
	app.run(debug=True)  # will run locally http://127.0.0.1:5000/


