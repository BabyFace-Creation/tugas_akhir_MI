# Real Time Human Detection, Counting, And Tracking

# imported necessary library
from functools import total_ordering
from tkinter import *
from centroidtracker import CentroidTracker
import tkinter as tk
import tkinter.messagebox as mbox
from PIL import ImageTk, Image
import cv2
import numpy as np
import imutils
import argparse
import matplotlib.pyplot as plt
from fpdf import FPDF
import datetime

protopath = "MobileNetSSD_deploy.prototxt"
modelpath = "MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

#Tracker
tracker = CentroidTracker(maxDisappeared=10, maxDistance=1800)


# Main Window & Configuration
window = tk.Tk()
window.title("Real Time Customer Detection & Counting")
window.geometry('1000x700')

# top label
start1 = tk.Label(text = "REAL-TIME-CUSTOMER\nDETECTION  &  COUNTING", font=("Arial", 50,"underline"), fg="magenta") # same way bg
start1.place(x = 70, y = 10)

# function defined to start the main application
def start_fun():
    window.destroy()

# created a start button
Button(window, text="▶ START",command=start_fun,font=("Arial", 25), bg = "orange", fg = "blue", cursor="hand2", borderwidth=3, relief="raised").place(x =130 , y =570 )

# image on the main window
path1 = "Images/front3.png"
img2 = ImageTk.PhotoImage(Image.open(path1))
panel1 = tk.Label(window, image = img2)
panel1.place(x = 90, y = 250)

# image on the main window
path = "Images/front1.png"
img1 = ImageTk.PhotoImage(Image.open(path))
panel = tk.Label(window, image = img1)
panel.place(x = 380, y = 180)

exit1 = False
# function created for exiting from window
def exit_win():
    global exit1
    if mbox.askokcancel("Exit", "Do you want to exit?"):
        exit1 = True
        window.destroy()

# exit button created
Button(window, text="❌ EXIT",command=exit_win,font=("Arial", 25), bg = "red", fg = "blue", cursor="hand2", borderwidth=3, relief="raised").place(x =680 , y = 570 )

window.protocol("WM_DELETE_WINDOW", exit_win)
window.mainloop()

if exit1==False:
    # Main Window & Configuration of window1
    window1 = tk.Tk()
    window1.title("Real Time Costumer Detection & Counting")
    window1.geometry('1000x700')


    def argsParser():
        arg_parse = argparse.ArgumentParser()
        arg_parse.add_argument("-v", "--video", default=None, help="path to Video File ")
        arg_parse.add_argument("-i", "--image", default=None, help="path to Image File ")
        arg_parse.add_argument("-c", "--camera", default=False, help="Set true if you want to use the camera.")
        arg_parse.add_argument("-o", "--output", type=str, help="path to optional output video file")
        args = vars(arg_parse.parse_args())
        return args

    # ---------------------------- camera section ------------------------------------------------------------
    def camera_option():
        # new window created for camera section
        windowc = tk.Tk()
        windowc.title("Human Detection from Camera")
        windowc.geometry('1000x700')

        max_count3 = 0
        framex3 = []
        county3 = []
        max3 = []
        avg_acc3_list = []
        max_avg_acc3_list = []
        array_person3 = []
        rects = []
        obj_id_list = []
        lpc_counts = 0
        max_acc3 = 0
        max_avg_acc3 = 0
        total = 0

        def non_max_suppression_fast(boxes, overlapThresh):
            try:
                if len(boxes) == 0:
                    return []
                if boxes.dtype.kind == "i":
                    boxes = boxes.astype("float")
                
                pick = []

                x1 = boxes[:, 0]
                y1 = boxes[:, 1]
                x2 = boxes[:, 2]
                y2 = boxes[:, 3]

                area = (x2 - x1 + 1) * (y2 - y1 + 1)
                idxs = np.argsort(y2)

                while len(idxs) > 0:
                    last = len(idxs) - 1
                    i = idxs[last]
                    pick.append(i)

                    xx1 = np.maximum(x1[i], x1[idxs[:last]])
                    yy1 = np.maximum(y1[i], y1[idxs[:last]])
                    xx2 = np.minimum(x2[i], x2[idxs[:last]])
                    yy2 = np.minimum(y2[i], y2[idxs[:last]])

                    w = np.maximum(0, xx2 - xx1 + 1)
                    h = np.maximum(0, yy2 - yy1 + 1)

                    overlap = (w * h) / area[idxs[:last]]

                    idxs = np.delete(idxs, np.concatenate(([last],
                                                        np.where(overlap > overlapThresh)[0])))
                return boxes[pick].astype("int")
            except Exception as e:
                print("Exception occurred in non_max_suppression : {}".format(e))

        # function defined to open the camera
        def open_cam():
            global max_count3, framex3, county3, max3, avg_acc3_list, max_avg_acc3_list, array_person3, obj_id_list, lpc_counts, rects, max_acc3, max_avg_acc3, total
            max_count3 = 0
            framex3 = []
            county3 = []
            max3 = []
            avg_acc3_list = []
            max_avg_acc3_list = []
            array_person3 = []
            rects = []
            obj_id_list = []
            lpc_counts = 0
            max_acc3 = 0
            max_avg_acc3 = 0
            total = 0

            args = argsParser()

            info1.config(text="Status : Opening Camera...")
            # info2.config(text="                                                  ")
            mbox.showinfo("Status", "Opening Camera...Please Wait...", parent=windowc)
            # time.sleep(1)

            writer = None
            if args['output'] is not None:
                writer = cv2.VideoWriter(args['output'], cv2.VideoWriter_fourcc(*'MJPG'), 10, (600, 600))
            if True:
                detectByCamera(writer)

        # function defined to detect from camera
        def detectByCamera(writer):
            global max_count3, framex3, county3, max3, avg_acc3_list, max_avg_acc3_list, array_person3, obj_id_list, lpc_counts, rects, max_acc3, max_avg_acc3, total
            max_count3 = 0
            framex3 = []
            county3 = []
            max3 = []
            avg_acc3_list = []
            max_avg_acc3_list = []
            array_person3 = []
            rects = []
            obj_id_list = []
            lpc_counts = 0
            max_acc3 = 0
            max_avg_acc3 = 0
            total = 0

            # function defined to plot the people count in camera
            def cam_enumeration_plot():
                plt.figure(facecolor='orange', )
                ax = plt.axes()
                ax.set_facecolor("yellow")
                plt.plot(framex3, county3, label="Human Count", color="green", marker='o', markerfacecolor='blue')
                plt.plot(framex3, max3, label="Max. Human Count", linestyle='dashed', color='fuchsia')
                plt.xlabel('Time (sec)')
                plt.ylabel('Human Count')
                plt.legend()
                plt.title("Enumeration Plot")
                plt.get_current_fig_manager().canvas.set_window_title("Plot for Camera")
                plt.show()

            def cam_accuracy_plot():
                plt.figure(facecolor='orange', )
                ax = plt.axes()
                ax.set_facecolor("yellow")
                plt.plot(framex3, avg_acc3_list, label="Avg. Accuracy", color="green", marker='o', markerfacecolor='blue')
                plt.plot(framex3, max_avg_acc3_list, label="Max. Avg. Accuracy", linestyle='dashed', color='fuchsia')
                plt.xlabel('Time (sec)')
                plt.ylabel('Avg. Accuracy')
                plt.title('Avg. Accuracy Plot')
                plt.legend()
                plt.get_current_fig_manager().canvas.set_window_title("Plot for Camera")
                plt.show()

            def cam_gen_report():
                pdf = FPDF(orientation='P', unit='mm', format='A4')
                pdf.add_page()
                pdf.set_font("Arial", "", 20)
                pdf.set_text_color(128, 0, 0)
                pdf.image('Images/Crowd_Report-1.png', x=0, y=0, w=210, h=297)

                pdf.text(132, 155, str(max_count3))
                pdf.text(107, 168, str(objectId+1))


                pdf.output('Crowd_Report.pdf')
                mbox.showinfo("Status", "Report Generated and Saved Successfully.", parent = windowc)

            cap = cv2.VideoCapture(1)

            fps_start_time = datetime.datetime.now()
            fps = 0
            total_frames = 0 
            
            while True:
                ret, frame = cap.read()
                frame = imutils.resize(frame, height=600,)
                total_frames = total_frames + 1

                (H, W) = frame.shape[:2]

                blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

                detector.setInput(blob)
                person_detections = detector.forward()
                rects = []
                for i in np.arange(0, person_detections.shape[2]):
                    confidence = person_detections[0, 0, i, 2]
                    if confidence > 0.5:
                        idx = int(person_detections[0, 0, i, 1])

                        if CLASSES[idx] != "person":
                            continue

                        person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                        (startX, startY, endX, endY) = person_box.astype("int")
                        rects.append(person_box)

                boundingboxes = np.array(rects)
                boundingboxes = boundingboxes.astype(int)
                rects = non_max_suppression_fast(boundingboxes, 0.3)

                objects = tracker.update(rects)
                for (objectId, bbox) in objects.items():
                    x1, y1, x2, y2 = bbox
                    x1 = int(x1)
                    y1 = int(y1)
                    x2 = int(x2)
                    y2 = int(y2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    text = "ID: {}".format(objectId)
                    cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

                    if objectId not in obj_id_list:
                        obj_id_list.append(objectId)

                fps_end_time = datetime.datetime.now()
                time_diff = fps_end_time - fps_start_time
                if time_diff.seconds == 0:
                    fps = 0.0
                else:
                    fps = (total_frames / time_diff.seconds)

                fps_text = "FPS: {:.2f}".format(fps)

                cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

                lpc_counts = len(objects)

                if (lpc_counts > max_count3):
                    max_count3 = lpc_counts

                lpc_txt = "Person : {}".format(lpc_counts)
                cv2.putText(frame, lpc_txt, (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

                cv2.imshow("Human Detection from Camera", frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break

            cap.release()
            info1.config(text="                                                  ")
            # info2.config(text="                                                  ")
            info1.config(text="Status : Detection & Counting Completed")
            # info2.config(text="Max. Human Count : " + str(max_count3))
            cv2.destroyAllWindows()

            #Button(windowc, text="Enumeration\nPlot", command=cam_enumeration_plot, cursor="hand2", font=("Arial", 20),bg="orange", fg="blue").place(x=100, y=530)
            #Button(windowc, text="Avg. Accuracy\nPlot", command=cam_accuracy_plot, cursor="hand2", font=("Arial", 20),bg="orange", fg="blue").place(x=700, y=530)
            Button(windowc, text="Generate Report", command=cam_gen_report, cursor="hand2", font=("Arial", 20),bg="gray", fg="blue").place(x=325, y=550)

        lbl1 = tk.Label(windowc, text="DETECT  FROM\nCAMERA", font=("Arial", 50, "underline"), fg="brown")  # same way bg
        lbl1.place(x=230, y=20)

        Button(windowc, text="OPEN CAMERA", command=open_cam, cursor="hand2", font=("Arial", 20), bg="light green", fg="blue").place(x=370, y=230)

        info1 = tk.Label(windowc, font=("Arial", 30), fg="gray")  # same way bg
        info1.place(x=100, y=330)
        # info2 = tk.Label(windowc, font=("Arial", 30), fg="gray")  # same way bg
        # info2.place(x=100, y=390)

        # function defined to exit from the camera window
        def exit_winc():
            if mbox.askokcancel("Exit", "Do you want to exit?", parent = windowc):
                windowc.destroy()
        windowc.protocol("WM_DELETE_WINDOW", exit_winc)


    # options -----------------------------
    lbl1 = tk.Label(text="OPTION", font=("Arial", 50, "underline"),fg="brown")  # same way bg
    lbl1.place(x=340, y=20)

    # image on the main window
    pathi = "Images/image1.jpg"
    imgi = ImageTk.PhotoImage(Image.open(pathi))
    paneli = tk.Label(window1, image = imgi)
    paneli.place(x = 90, y = 110)

    # image on the main window
    pathv = "Images/image2.png"
    imgv = ImageTk.PhotoImage(Image.open(pathv))
    panelv = tk.Label(window1, image = imgv)
    panelv.place(x = 700, y = 260)# 720, 260

    # image on the main window
    pathc = "Images/image3.jpg"
    imgc = ImageTk.PhotoImage(Image.open(pathc))
    panelc = tk.Label(window1, image = imgc)
    panelc.place(x = 90, y = 415)

    # created button for all option
    Button(window1, text="DETECT FROM CAMERA ➡",command=camera_option, cursor="hand2", font=("Arial", 30), bg = "light green", fg = "blue").place(x = 110, y = 300)

    # function defined to exit from window1
    def exit_win1():
        if mbox.askokcancel("Exit", "Do you want to exit?"):
            window1.destroy()

    # created exit button
    Button(window1, text="❌ EXIT",command=exit_win1,  cursor="hand2", font=("Arial", 25), bg = "red", fg = "blue").place(x = 440, y = 600)

    window1.protocol("WM_DELETE_WINDOW", exit_win1)
    window1.mainloop()

