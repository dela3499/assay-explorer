# Documentation

## Uploading new data

Summary:

1. Create `data.zip` file like [example]().

2. Upload [here](http://45.55.10.127:8080/tree/add-data).

3. Add it to the database [here](http://45.55.10.127:8080/notebooks/assay-explorer/reorg/upload-new-data.ipynb).

To analyze the data from an experiment, you'll need to upload the raw data from the microscope, the layout of the plates, and some additional information about the experiment (who was responsible, when the data was collected, etc.)

You'll need to wrap all this data up in CSV files in a zip file like [this example](). There's a nicely-formatted Excel spreadsheet there that you can use to generate the CSV files as well.

1. Raw microscope data
Each plate should have a separate file.

2. Plate layouts
You'll need at least one plate layout. A single layout can be applied to many different plates. Check out the example above to see the format.

3. Additional Info
For every plate in your experiment, you'll create one row in a CSV file called metadata.csv. You'll include the following information:
 - **Plate File** Name of raw data file from microscope
 - **Layout File** For every plate, you'll need to specify which layout file describes it.
 - **Assay** Put the name of the assay here
 - **Image Collection Date** Use format like 03-29-2015
 - **Investigator** - you can include the initials of anyone responsible for the data
 - **Magnification** If using a 60X objective, just write 60
 - **Image Analysis Recipe** Name of the measurement step in MetaXpress
 - **Experiment Name** Put a descriptive name here

---

As you'll see in the example zip file, all this information is arranged like so in a file called `data.zip`:

  /data.zip
    metadata.csv
    /Layouts
      layout1.csv
      layout2.csv
    /Plates
      plate1.csv
      plate2.csv
      plate3.csv

Once you've got your zip file ready, you can go [here](http://45.55.10.127:8080/tree/add-data) and click upload in the upper-right corner of the screen. If there's already a `data.zip` file there, you should delete it by clicking the checkbox beside the file and the clicking the red trash can that appears near the top of the screen.

If you've got a fair amount of data, then this upload process could take a minute or two.

Once your data is uploaded, go [here](http://45.55.10.127:8080/notebooks/assay-explorer/reorg/upload-new-data.ipynb) to add it to the database. As always, click `Results Only` button and also click `Cell -> Run All` in the menu.

Then, click `Check Data`. This checks that all the relevant files are present, and then it combines all the information in the zip file into one csv that can be added to the database. This step takes 2-3 minutes.

Then click `Save Data` to add your data to the database.

If you make a mistake (like a typo in a plate layout), you can delete the dataset from the database using the last section and follow the process above to reupload it.

<!--
## Labelling Cell Phase


## Visualizing Data
-->
