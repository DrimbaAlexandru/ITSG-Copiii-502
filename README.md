# The Kids from Room 502

This application is an automatic assistant that 
comes to the aid of a medical student. It was developed within
the course _ITSG_ by the team __The Kids from Room 502__, 
consisting of __Cotutiu Ioana, Drimba Alexandru, Duma Iulia 
Ana-Maria, Filipciuc Andreea, Gherghel Denisa-Maria__.

## Presentation and documentation
The presentation can be accessed here: https://prezi.com/b94jstr1yfpd/?utm_campaign=share&utm_medium=copy .
The documentation can be found in this repo, as __ITSG_Documentation.pdf__

## Running the app
In order to open the app, the file __app.py__ must be run. 


## Functionalities
1. The user can upload the medical image of a heart, which the system will then display;
2. The system can display a 2D plot for each of the image's axis (axial, coronal, and saggital), and the user can scroll along any of the axis;
3. The system can generate labels for an uploaded medical image; the generated labels will depict the ventricular myocardium, blood pool, and the background;
4. The user can view the original medical image with the generated labels overlayed, having control over the opacity of the labels as a whole;
5. The user can view a rendered 3D view of the generated labels with the possibility of interacting with it;
6. The user can save both the generated labels and the 3D rendered view to a path of his choice;
7. The user can choose what U-Net model to be used for the generation of the
labels: 2D U-Net or 3D U-Net (the improved one).


## Logs
For easier debugging, the messages printed in the console start with 
_INFO, WARN_ and _ERROR_. As further improvements, at least for 
_WARN_ and _ERROR_, special dialogs can be added to inform the 
user of the occurrence of these events.
