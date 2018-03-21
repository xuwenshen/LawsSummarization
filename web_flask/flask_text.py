import os
import numpy as np
import tensorflow as tf
import json
from flask import Flask, render_template, session, redirect, url_for
from flask.ext.wtf import Form
from wtforms import TextAreaField,SubmitField
from flask.ext.script import Manager
from flask.ext.bootstrap import Bootstrap # boostrap 扩展
from flask.ext.moment import Moment
import re
import h5py
import random
from wtforms.validators import Required

from Generation import Generate


app = Flask(__name__)
app.config['SECRET_KEY'] = 'you-will-never-guess'

manager = Manager(app)
bootstrap = Bootstrap(app)
moment = Moment(app)
generate = Generate()


class InputForm(Form):
    input_text = TextAreaField(u'',validators=[Required()],render_kw={"rows": 12})
    text_submit = SubmitField(u'提交')
    


class Sample(object):
    def __init__(self):
        self.ifile = open('trick_cases')
        self.current_sample = dict()
        
    def next(self):
        line = self.ifile.readline()
        
        line = json.loads(line)
        self.current_sample = line
        
        to_print = ''

        for i in range(len(line['tag'])):
            to_print += line['tag'][i] + ': ' + line['source'][i] + '\n\n'
        return to_print
    
    
sample = Sample()


@app.route('/random')
def random_text():
    return sample.next()


@app.route('/', methods=['GET','POST'])
def index():
    
    form = InputForm()

    
    if form.validate_on_submit():
        
        if form.text_submit.data:
            raw_text = form.input_text.data
            tx = generate.query(raw_text)
            
            return render_template('index.html',
                                   form=form,
                                   result_text=tx['prediction_tx'],
                                   ground_truth='')
        
        else:
            
            tx = generate.sample(sample.current_sample)
            
            return render_template('index.html',
                                   form=form,
                                   result_text=tx['prediction_tx'],
                                   ground_truth=sample.current_sample['reason'])
        
    
    
    return render_template('index.html',
                           form=form,
                           result_text='',
                           ground_truth='')

 



if __name__ == '__main__':
    app.run('192.168.1.97',debug=False,port=10007)
    manager.run()

