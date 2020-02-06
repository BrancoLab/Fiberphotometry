import cv2
import pandas as pd

def manually_define_rois(frame, n_rois, radius):
    """
        Lets user manually define the position of N circular ROIs 
        on a frame by clicking the center of each ROI
    """

    # Callback function
    def add_roi_to_list(event, x, y, flags, data):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(data[0], (x, y), data[1], (0, 255, 0), 2) 
            cv2.circle(data[0], (x, y), data[1], (0, 0, 255), 3)  
            return data[2].append((x, y))

    ROIs = []
    # Start opencv window
    cv2.startWindowThread()
    cv2.namedWindow('detection')
    cv2.moveWindow("detection",100,100)   
    cv2.imshow('detection', frame)

    # create functions to react to clicked points
    data = [frame, radius, ROIs]
    cv2.setMouseCallback('detection', add_roi_to_list, data)  

    while len(ROIs) < n_rois:
        cv2.imshow('detection', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows() 
    return ROIs


def split_blue_violet_channels(traces, blue_first=True):
    """
        Takes a dataframe with the signal for each fiber and splits the blue and violet frames. 
    """
    columns = traces.columns

    new_data = {}
    for col in columns:
        if blue_first:
            blue = traces[col].values[0:-1:2]
            violet = traces[col].values[1:-1:2]

        else:
            raise NotImplementedError
        if len(blue) != len(violet):
            raise ValueError

        new_data[str(col)+"_blue"] = blue
        new_data[str(col)+"_violet"] = violet
    return pd.DataFrame(new_data)