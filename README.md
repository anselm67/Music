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
pdmx query -o Staff16.cvs 'index==index' --score 'pages.*.staff_count < 16'
# Displays subset general statistics.
pdmx --csv subset.csv stats
# Train the staffer on that dataset.
staffer --log-file logs/staffer.log train -e 12 --use-sampler model_name
```

For the noter model, the same process applies:

```bash
# Select all scores that have systems of only one staff:
pdmx query -o System2.csv 'index==index' --score 'pages.0.systems.0.staff_count <= 2' --valid
```

TODO List:
- pdmx: add a validate command to filter out from the underlying csv any row that isn't completly made (eg that's missing a tokens file  or carries broken images).
```bash
  # pdmx --csv Staff16.csv validate -o ValidStaff16.csv
```
- Add a validate command to staffer that gives real metrics on full set validation; Requires a new DataLoader to pick from the samples not used during training.
- Simplify the network output so staff becomes two coordinates only (top, bottom) derive other coordinates from the system
- import editor from projects/Staffer or projects/OMR
- Remove bar prediction from the model
- move staffer model, dataset into a staffer package

Pending fixes:
- The tokenizer should check the length of the first bars against the metric and decide based on that where the number 1 falls.
- In mxl/1/1/Qmb24DrN1PECaithcX1YzvEquFF4QDGLDoeEZUjJsF7Etk.mxl, there is a mutli-bar rest at the 
end of the first page; The tokenizer needs to take it apart into separate bars.
- In mxl/14/10/QmWAGXyEP8SJRRRPSy5jpFvX9MRGPqPUuHkUay19hAy8wM.mxl the rendering is missing the first few bars and is therefore out of sync.
- In /mxl/3/6/Qmd7UQFcdQg8fjqqCkJPHkc2N4PqQEkx6vh5sxqchozJu8.mxl the bar count mismatches likely because they are some invisible bars at the beginning of the svg file that the LayoutExtractor counts (it shouldn't).
- In mxl/3/47/QmdUzkYhyPhs5b5TNvGmT2fpc1CwLdjkxFrWvGE5WCFECa.mxl right after the second page break, bars have leading dots in spine.
- In mxl/1/11/QmbbGKtZ9G6DkWxvSeU516c1ktWiFJmEbHGmR3JFtLAPyC.mxl, the tokenized version of the file has a spine count mismatch due to a spine branch.

