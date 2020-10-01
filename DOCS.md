# Building the Documentation

To create the html version of the documentation do the following. After you have [installed the package](docs/library/getting_started.md), install the the additional requirements using 
```bash
> pip install -r requirements-docs.txt
```
Then run 
```bash
> mkdocs serve 
```
to start a local server hosting the site, or run
```bash
> mkdocs build 
```
to build a static html version of the documentation (which can be found in the folder `sites`). Both of the above uses the `mkdocs.yml` file to build the documentation. Please see the [mkdocs documentation](https://www.mkdocs.org) for additional information.
 
