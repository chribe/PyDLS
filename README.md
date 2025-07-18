# PyDLS
## general comments
The package is written to analyse DLS data. 
The data has to be provided in a folder, in which each subfolder contains the sets of one measurement.
The script merges all measurements performed on one angle in one subfolder together, as long as the sample temperature in within delta T =1.5K.
If several subfolders are included, the samples are treated independently.
Currently, only data from ALV DLS are imported.
Several models for the correlation functions are implemented.

% Todo: list models

The q dependence of the different fit parameters can be approximated in a second step by custom lambda functions. The sample dependencies can then be displayed in a following step.
Via the GUI, analysis results can be saved as exel file and HDF file. Automatically compiled Summaries based on LaTeX can be created. The HDF Files can be pushed to Zenodo to publish them and to obtain a doi.

Software has been tested on Linux.
## Script based analysis
For Script based analysis launch the script "DLS_Analysis_clean.py" after modifying the corresponding paths.

## GUI
To start the gui, launch it via "python gui.py"
### Basic Steps:
1. _Load Data_: Select the folder which contains the subfolders for for the different samples. Once loaded, the correlation function of the first sample is displayed. By using the sliders, it is possible to scroll through the different samples and q values.
2. _Fit the correlation function_: Choose the adapted model(s). Once the fit is done, they can be displayed by ticking the boxes on the right hand side under the plotting area.
3. _Display the _q_ dependence of the parameters_: By choosing the bullet point "q-dependence" in the Plot results section, The plotting area changes to a second view, where the $q$ dependence can be displayed as a function of q or q^2. If only one fit parameter is selected, the q dependence can be fitted via the button "Fit q dependence" which is only visible in the "q-dependence" option. A new window asks for a lambda function which should be used to fit the q dependence. Once entered, the Fit parameters have to be obtained. Its start values and limits can be set in the table. Confirming the entry with "Done" performs the fit on all Samples for the displayed parameter. The process can be repeated for different parameters with the same or also different fit models. With the slider present, the q dependence of the different samples can be investigated.
4. _dependence of the fit parameters from the q dependence_: By choosing the last option in the Plot results section, the different fit parameters can be investigated as a function of the sample. Currently, the samples can only be displayed as a function of the index.

$ToDo: implement option to show it as a function of time, temperature, custom input

5. _Save Results_: As a last step, the Data can be saved as HDF files and Exel Files. Summaries can be created showing all the different fits in a pdf per sample via "Compile Summaries". For this step, XeLaTeX has to be installed. Data can also be uploaded to Zenodo.org to obtain a DOI. See following paragraph for more details.

### Zenodo
To publish the analysed data set on Zenodo.org, it is first necessary, to create first a personal account. In this account, create then a personal access token under "My account/Applications" with the scopes "Deposit:actions" and "deposit:write".
Save the obtained Access Token in the same folder in a text file called "PyDLSsettings.txt". The entry should look like "AccessToken:__YOUR__ACCESS__TOKEN__".
The data has to be first saved as HDF file before being pushed to Zenodo!

### Changing color scheme 
The color scheme of the gui can be adapted by  "Settings>Change Color Scheme". It swithes between a light mode, dark mode and a mode in which the color scheme can be customly defined. Fot the last option, it is necessary to add a file called "guicolors.txt". The file must contain 6 lines with one entry each defining the following colors:
1. main color (default white #FFFFFF)
2. color of plotting background (default grey #D9D9D9)
3. color of button not hovered (default green #4CAF50)
4. color of button hovered (default dark green #45a049)
5. color of boxes (default blue #4D00FF)
6. color of text (default black #000716)

The files *ilt.py* and *ldp.py* are for the Contin-like analysis and are modified based on https://github.com/caizkun/pyilt.
