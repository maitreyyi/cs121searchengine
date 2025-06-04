from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Homepage with search UI
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to handle search
@app.route('/search')
def search():
    query = request.args.get('q')
    # You can plug in your own search function here
    results = ["Result for: " + query]
    return jsonify(results)

# Endpoint for autocomplete
@app.route('/autocomplete')
def autocomplete():
    term = request.args.get('term', '').lower()
    suggestions = [s for s in ['apple', 'amazon', 'alpha', 'zebra', 'zoom'] if s.startswith(term)]
    return jsonify(suggestions)