import inference
from flask import Flask
from flask import render_template,request,url_for, flash, redirect
messages = [{'title': ' ', 'sentiment': ' '}]
app = Flask(__name__)
app.config['SECRET_KEY'] = 'd91bfb0e51dcc99ee2b6ed0acaf8b7e016ea796c855fcd8f'
@app.route('/')
def index():
    return render_template('index.html', messages=messages)

@app.route('/create/', methods=('GET', 'POST'))
def create():
    messages = [{'title': ' ', 'sentiment': ' '}]
    if request.method == 'POST':
        title = request.form['title']

        if not title:
            flash('Title is required!')
        else:
                model= inference.get_model()
                mov= inference.Movie(str(title))
                title,rev,flag= mov.get_reviews()
                if flag==0:
                    x= mov.get_sentiment(rev,model)
                    messages=[{'title': str(title), 'sentiment':str(x)}]

                else:
                    flash('URL not found... Check movie title')
        
        return render_template('create.html', messages= messages)

    return render_template('create.html', messages= messages)
 
@app.route('/about/')
def about():
    return render_template('about.html')

if __name__=='__main__':
   app.run()
