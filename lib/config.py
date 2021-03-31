## Temporary Path Storage for simple plug-in-play development

DataInputDirectory = '/home/jeffmur/data/mdcd/user_by_month'
"""
Note: Used with os.walk which assumes it is a directory (hence no tailing backslash)
"""

DataOutputDirectory = '/home/jeffmur/data/mdcd/'
"""
Used for exporting data (as images or csvs)
"""

GitPath = '/home/jeffmur/dev/mdcd/'
"""
Project Directory
"""

CondaEnviornment = '/home/jeffmur/anaconda3/condabin/conda'

datasetHeaders = ['Index', 'UID' , 'Date' , 'Time' , 'Latitude' , 'Longitude']
"""
Specific to dataset in question
"""

# TODO: ADD MORE AS NECESSARY
## @alex please advise on tools for Setting up Enviornment & python-dotenv configuration
