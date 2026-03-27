# Python tools for managing Music datasets for OMR.

This repository has a set of tools to manage, maintain and expand existing datasets for the purpose of training OMR models. The toolkit consists of the following packages:
- kern: parsing humdrum kern files,
- midi: parse, create and edit midi files,
- imslp: grab sheet music from the IMSLP site,
- editor: create and edit annotatipons layered on top of existing sheet music, eg page structure informations such as staves.

Dataset sources:
- PDMX: https://zenodo.org/records/14648209
    Requires downloading multiple files, at least PDMX.csv, mxl.tar.gz and metadata.tar.gz
    
Examples:

To extract a reasonable subset from PDMX, you can use something like this:

```bash
# Select scores that have all pages rendering less than 16 staves.
pdmx query -o subset.cvs 'index==index' --score 'pages.*.staff_count < 16'
# Displays subset general statistics.
pdmx --csv subset.csv stats
```
TODO: import imslp from projects/OMR
TODO: import editor from projects/Staffer or projects/OMR
