# pyfin

Computational finance methods in Python.

### Install

Install the module (for development) by running

```
python -m pip install -e .
```

and then use by importing `pyfin`. See the `report` notebooks for examples.

### Export notebooks

From the `pyfin/report` directory, export the report by running:

```
rm report.ipynb -f && nbmerge -r -o report.ipynb && python -m jupyter nbconvert report.ipynb --to webpdf --embed-images
```
