"""Execute every notebook cell, save outputs even on failure, and show progress."""
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

base=Path(__file__).resolve().parents[1]
path=base/'VaDE_CIFAR10.ipynb'
nb=nbformat.read(path,as_version=4)

def completed(cell, cell_index, **kwargs):
    print(f'Cell {cell_index+1}/{len(nb.cells)} finished',flush=True)
    for output in cell.get('outputs',[]):
        if output.output_type=='stream': print(output.text[-2500:],flush=True)
    nbformat.write(nb,path)

client=NotebookClient(nb,timeout=None,kernel_name='python3',
                      resources={'metadata':{'path':str(base)}},
                      on_cell_executed=completed)
try:
    client.execute()
finally:
    nbformat.write(nb,path)
print('Completed:',path,flush=True)
