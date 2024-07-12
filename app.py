# backend.py
from flask import Flask, request, jsonify
from src.customer_assistance_Agent import CustomerAssistanceAgent

app = Flask(__name__)

agent = CustomerAssistanceAgent()

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_input = data.get("user_input")
    
    if user_input:
        response = agent.query_with_prefix(user_input)
        return jsonify({"response": response})
    return jsonify({"response": "Invalid input"}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
