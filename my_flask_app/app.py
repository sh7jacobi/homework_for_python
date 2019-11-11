from flask import Flask, render_template, request, url_for
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/data.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    level = db.Column(db.INTEGER, unique=False, nullable=False)

    def __repr__(self):
        return '<User %r>' % self.username

db.create_all()

@app.route('/', methods=['POST', 'GET'])
def student():
    if request.method == 'POST':
        name  = request.form['Name']
        level = request.form['level']
        db.session.add(User(username=name, level=level))
        db.session.commit()
    return render_template('student.html')

@app.route('/result',methods = ['GET'])
def result():
    return render_template('result.html', data=User.query.all())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
