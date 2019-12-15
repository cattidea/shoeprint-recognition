import os
import subprocess
import time
import webbrowser

from config_parser.config import DOCS

docs_dir = DOCS['docs_dir']
port = DOCS['port']
host = 'localhost'


def docs_dev():
    shell = os.name == "nt"
    p = subprocess.Popen(["python3", "-m", "http.server",
                          str(port)], shell=shell, cwd=docs_dir)
    webbrowser.open(f'http://{host}:{port}')
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            p.terminate()
            break
