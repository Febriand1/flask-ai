from flask import Flask
from api.index import welcome, chat
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}})

@app.route('/', methods=['GET'])
def index():
    return welcome()

@app.route('/chat', methods=['POST'])
def chatbot():
    return chat()

if __name__ == '__main__':
    app.run()