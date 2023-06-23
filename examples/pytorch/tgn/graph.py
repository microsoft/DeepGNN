import urllib.request
import tempfile
import os

def prepare_data(name:str):
    # working_dir = tempfile.TemporaryDirectory()
    os.mkdir(f"/tmp/temporal_{name}")
    file = f"/tmp/temporal_{name}/{name}.csv"
    urllib.request.urlretrieve(
        f"http://snap.stanford.edu/jodie/{name}.csv", file # working_dir.name
    )
