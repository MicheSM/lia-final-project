from . import app

# Define a simple route for testing
@app.route('/')
def hello_world():
    return 'Hello!'