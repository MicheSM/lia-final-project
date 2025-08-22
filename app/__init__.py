from flask import Flask

# Initialize the Flask application
app = Flask(__name__)

# Import routes after app creation to avoid circular imports
from .routes import *