import os
import sys
from streamlit.web import cli as stcli

if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, 'app.py')
    sys.argv = ["streamlit", "run", filename]
    sys.exit(stcli.main())