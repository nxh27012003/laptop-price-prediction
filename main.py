
import sys
import os

path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, path)
from src.app.app import gui_app

if __name__=='__main__':
    gui_app()