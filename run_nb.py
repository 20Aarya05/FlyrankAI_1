import nbformat
from nbclient import NotebookClient

with open('work/notebooks/w05_model.ipynb') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name='python3')
client.execute()

with open('work/notebooks/w05_model.ipynb', 'w') as f:
    nbformat.write(nb, f)
