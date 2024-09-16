from flask import request, jsonify
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modul import reply

AnswerList = []
AccuracyList = []
QuesionList = []

def welcome():
    return jsonify({'message': 'Welcome to the API'}), 200

def chat():
    try:
        data = request.get_json()
        message = data.get('message')
 
        if not message:
            return jsonify({"error": "No message provided"}), 400

        return_message, status, dec_outputs, accuracy = reply.botReply(message)
        AnswerList.append(return_message)
        AccuracyList.append(float(accuracy))
        QuesionList.append(message)
        
        response = {
            "isBot" : True, 
            "ITeung": return_message,
            "status": status,
            "accuracy": float(accuracy),
            # 'dec_outputs': dec_outputs.tolist() if isinstance(dec_outputs, (list, np.ndarray)) else dec_outputs
        }

        return jsonify(response), 200
    
    except FileNotFoundError as e:
        return jsonify({"error": f"File not found: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500
