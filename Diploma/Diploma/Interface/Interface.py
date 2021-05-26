from tkinter import *
from tkinter import messagebox as box
from PIL import ImageTk, Image
import imageio
import threading
import imutils
import cv2
import os
import time
import queue
import numpy as np
import cntk as C

import Estimation.Estimator as estima

from ExitWindow import *
from Dialog import *



class PhotoBoothApp:
    def __init__(self, video_capture, output_path):
        self.video_capture = video_capture
        self.output_path = output_path
        self.width = 1920
        self.height = 800
        self.max_width = 640
        self.shape_to_resize = (224, 224)
        self.coef_normalize = 127.5

        self.frame = None
        self.number_frames = 30 # 30 * 1 = 30 frames in 1 sec
        self.cnt_frames = 0
        self.interval = 1
        self.results = []

        self.root = Tk()
        self.root.title('Saberfighting teacher')
        self.root.geometry('1280x720')

        self.video_window = None
        self.label_window = None
        self.video_name = ""

        self.path_vertical_hit = r'D:\mine\diploma\Dataset\Etalon\vertic.mp4'
        self.path_horizontal_hit = r'D:\mine\diploma\Dataset\Etalon\horiz.mp4'


        self.root.configure(background = 'black')

        label = Label(self.root, text = 'Saberfighting teacher', fg = 'light blue',
                      bg = 'black', font = ('Verdana', 50, 'bold'), height = 2)
        label.pack(side = TOP)

        gridFrame = Frame(self.root, bg='black') # New frame to store buttons

        # Grid uses sticky instead of anchor, but in this scenario it is not really necessary
        # I have left it in case you need it for some other reason

        Jam = Button(gridFrame, text = 'START', font=("Verdana", 20, "bold"),
                     width = 25, height = 4, bg="light blue", fg='navy',
                     command=self.start_video)
        Jam.grid(row=2, column=1, pady = 10, padx = 25, sticky=W) 
        Roses = Button(gridFrame, text = 'EXIT', font=("Verdana", 20, "bold"),
                       width = 25, height = 4, bg="light blue", fg='navy',
                       command=lambda:[self.root.destroy()])
        Roses.grid(row=3, column=1, pady = 10, padx = 25, sticky=W)

        Madness = Button(gridFrame, text = 'VERTICAL HIT VIDEO', font=("Verdana", 20, "bold"),
                         width = 25, height = 4, bg="light blue", fg='navy',
                         command= lambda: self.action(self.path_vertical_hit, 'VERTICAL HIT VIDEO'))
        Madness.grid(row=2, column=2, pady = 10, padx = 25, sticky =N)

        Pistols = Button(gridFrame, text = 'HORIZONTAL HIT VIDEO', font=("Verdana", 20, "bold"), 
                         width = 25, height = 4, bg="light blue", fg='navy',
                         command= lambda: self.action(self.path_horizontal_hit, 'HORIZONTAL HIT VIDEO'))
        Pistols.grid(row=3, column=2, pady = 10, padx = 25, sticky =N)


        gridFrame.pack(side=TOP) 
        # Place it a the bottom or top or wherever you want it to go
        self.root.protocol('WM_DELETE_WINDOW', self.exit_mode)
        self.root.mainloop()


    def exit_mode(self):
        dialog = Exit_window(self.root)
        my_message = 'Do you want to exit?'
        return_value = dialog.go(message = my_message)
        if return_value:
            self.root.destroy()


    def exit_video_window(self):
        dialog = Exit_window(self.video_window)
        my_message = 'Do you want to exit?'
        return_value = dialog.go(message = my_message)
        if return_value:
            self.video_window.destroy()


    def action(self, path_to_video, name_action):
        self.video_window = Toplevel()
        self.video_window.title(name_action)
        # self.video_window.geometry('720x720')
        # self.video_window.configure(background = 'black')
        self.video_name = path_to_video
        self.start_etalon_video(self.video_window)
        self.video_window.protocol('WM_DELETE_WINDOW', self.exit_video_window)


    def start_etalon_video(self, window):
        video = imageio.get_reader(self.video_name)
        label = Label(window)
        label.pack(side="top", fill="both", expand=True)
        btn = Button(window, text="OK", width = 25, height = 2, font=("Verdana", 16, "bold"),
                     bg="light blue", fg='navy', command=window.destroy)
        btn.pack(fill="both", expand=True, padx=10, pady=10)
        thread = threading.Thread(target=self.stream, args=(label, video))
        thread.daemon = 1
        thread.start()


    def stream(self, label, video):
        for image in video.iter_data():
            frame_image = ImageTk.PhotoImage(Image.fromarray(image))
            label.configure(image=frame_image)
            label.image = frame_image


    def start_video(self):
        self.video_window = Toplevel()
        self.video_window.title('Record action')
        self.video_window.geometry('1280x720')
        self.video_window.configure(background = 'black')

        self.label_window = tk.Label(self.video_window)
        self.label_window.pack()
        # timer
        times = 3
        label1 = Label(self.video_window, anchor=S, text="", font=('Helvetica', 50),
                          bg='white', fg='navy')
        label1.pack()
        while (times):
            label1.configure(text=str(times))
            label1.update()
            time.sleep(1)
            times -= 1
        label1.configure(text="GO!")
        label1.update()
        label1.destroy()
        self.webcam_video()
        self.classification()
        self.video_window.protocol('WM_DELETE_WINDOW', self.exit_video_window)
        self.video_window.mainloop()


    def webcam_video(self):
        # canvas = tk.Canvas (self.video_window, width = 640, height = 480, bg = "white")
        # canvas.pack ()
        results = []
        while(self.cnt_frames < self.number_frames):
            ret, frame = self.video_capture.read()

            if frame is None:
                raise ConnectionError

            if ((self.cnt_frames + 3) % 3) == 0:
                frame_resized = imutils.resize(frame, width=self.max_width)
                print(self.cnt_frames)
                self.results.append(frame_resized)

            self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # to RGB
            self.image = Image.fromarray(self.image) # to PIL format
            self.image = ImageTk.PhotoImage(self.image) # to ImageTk format

            self.label_window.imgtk = self.image
            self.label_window.configure(image=self.image)
            self.label_window.update()
            # Update image
            # self.video_window.create_image(0, 0, anchor=NW, image=self.image)

            # Repeat every 'interval' ms
            # self.label_window.after(self.interval, self.webcam_video)
            self.cnt_frames += 1
        self.video_capture.release() 


    def classification(self):

        model_softmax = C.functions.load_model(r'D:\mine\diploma\Models\Siamese\with augmentation\softmax_cnn_saberfighting.model')
        model_vertic = C.functions.load_model(r'D:\mine\diploma\Models\Siamese\with augmentation\vertic_triplet_saberfighting.model')
        model_horiz = C.functions.load_model(r'D:\mine\diploma\Models\Siamese\with augmentation\horiz_triplet_saberfighting.model')

        # outputs of softmax
        not_fencing = 0
        vertic_softmax = 1
        horiz_softmax = 2

        # labels for triplets
        vertic_bad = 2
        vertic_good = 3
        vertic_threshold = 2.1e-9

        horiz_bad = 4
        horiz_good = 5
        horiz_threshold = 2.5e-12

        vertic_anchor = np.array([-0.24746865, 0.03978111, 0.13912132, -0.01408572, 0.03013963, -0.06000096,
        -0.3303202, 0.1385255, 0.04360551, -0.05080905, -0.02672888, -0.11045513,
        0.10860564, -0.07108926, 0.05151974, -0.08624982], dtype=np.float32)
        horiz_anchor = np.array([ 0.00753562, -0.09878368, -0.0634775, 0.17690332, 0.11552318, 0.07101858,
        0.18993254, 0.02002085, -0.29981878, 0.0445806, 0.038764, -0.02385881,
        0.14823848, 0.27717984, 0.10368124, 0.13088007], dtype=np.float32)

        Estimator = estima.Pose_estimation_mode()

        video = []

        for frame in self.results:
            skelet_frame = Estimator.process(frame)
            im = cv2.resize(skelet_frame, self.shape_to_resize, interpolation=cv2.INTER_AREA)
            im = np.array(im, dtype=np.float32)
            im -= self.coef_normalize
            im /= self.coef_normalize
            # (channel, height, width)
            im = np.ascontiguousarray(np.transpose(im, (2, 0, 1)))
            video.append(im)

        video_ = np.stack(video, axis=1)
        softmax_pred = np.squeeze(model_softmax.eval(video_))
        softmax_label = np.argmax(softmax_pred)
        prediction = not_fencing
        if softmax_label == vertic_softmax: # saberfighting vertic hit
            vertic_triplet_pred = np.squeeze(model_vertic.eval(video_))
            vertic_pred_dist = np.sum(np.square(vertic_anchor - vertic_triplet_pred))
            prediction = vertic_good if vertic_pred_dist < vertic_threshold else vertic_bad
        elif softmax_label == horiz_softmax: # saberfighting horiz hit

            horiz_triplet_pred = np.squeeze(model_horiz.eval(video_))
            horiz_pred_dist = np.sum(np.square(horiz_anchor - horiz_triplet_pred))
            prediction = horiz_good if horiz_pred_dist < horiz_threshold else horiz_bad
        else:
            prediction = not_fencing

        self.reset()
        if (prediction == vertic_good or prediction == horiz_good):
            label1 = Label(self.label_window, anchor=S, text="Results are good. Keep going!", font=('Helvetica', 50),
                           bg='black', fg='light blue')
            label1.pack()
        elif (prediction == vertic_bad):
            label1 = Label(self.label_window, anchor=S, text="You should watch the video to be better at vertical hints!", font=('Helvetica', 50),
                           bg='black', fg='light blue')
            label1.pack()
            self.action(self.path_vertical_hit, 'VERTICAL HIT VIDEO')
        elif (prediction == horiz_bad):
            label1 = Label(self.label_window, anchor=S, text="You should watch the video to be better at horizontal hints!", font=('Helvetica', 50),
                           bg='black', fg='light blue')
            label1.pack()
            self.action(self.path_horizontal_hit, 'HORIZONTAL HIT VIDEO')
        else:
            label1 = Label(self.label_window, anchor=S, text="Results are very bad. Try again!", font=('Helvetica', 50),
                           bg='black', fg='light blue')
            label1.pack()
        btn = Button(self.label_window, text="Next", width = 25, height = 2, font=("Verdana", 16, "bold"),
                         bg="light blue", fg='navy', command=self.video_window.destroy)
        btn.pack(fill="both", expand=True, padx=10, pady=10)


    def reset(self):
        self.cnt_frames = 0
        #self.video_capture = cv2.VideoCapture(0)
        self.video_capture = cv2.VideoCapture(r'D:\mine\diploma\Dataset\Data\Raw\Vertic\Bad\vertic_bad_2.mp4')
        self.results = []


if __name__ == "__main__":
    #PhotoBoothApp(cv2.VideoCapture(0), r'')
    PhotoBoothApp(cv2.VideoCapture(r'D:\mine\diploma\Dataset\Data\Raw\Vertic\Bad\vertic_bad_2.mp4'), r'')
    