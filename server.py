from flask import Flask, render_template, request, Markup, flash, Markup, make_response
import pandas
import sys
import os
sys.path.append("./src")
from recommendation import *
from content_title_rec import *
# from summarizer import Summarizer
# import spacy # with pretrained model to recognize entities
# from spacy import displacy
from pathlib import Path
from Topic_modeling import *
# nlp = spacy.load("en_core_web_lg")
# Create the application object
app = Flask(__name__)
os.chdir('./data')
#df = pandas.read_csv('sum_ents.csv')
df = pandas.read_csv('further_summ_ents.csv')
os.chdir('../')


@app.route("/")
def home_page():   
	return render_template('index.html')


@app.route('/output', methods=['get', 'post'])
def recommendation_output():
	#
	# Pull input
	some_input = request.args.get('user_input')

	# Case if empty
	if some_input == "":
		return render_template("index.html", my_input=some_input,my_form_result="Empty")
	else:
		# some_output="Double check your web!!Something wrong!!"
		#some_output, out_df = get_close_docs(some_input, df)
		some_output, out_df = get_con_close_docs(some_input, df)
		output_dict = out_df.to_dict('records')
		columnNames = out_df.columns.values
	# some_number = 3
	some_image = "panda.gif"
	return render_template("index.html", my_input=some_input,
						   # my_output0=some_output[2], my_output1=some_output[3],
						   # my_output2=some_output[4], my_output3=some_output[5],my_output4=some_output[6],
						   # my_output5=some_output[7],
						   records=output_dict, colnames=columnNames,
						   #my_number=some_number,
						   my_img_name=some_image,
						   my_form_result="NotEmpty")


# start the server with the 'run()' method
if __name__ == "__main__":
	app.run(debug=True)  # will run locally http://127.0.0.1:5000/


