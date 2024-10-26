import cv2
import numpy as np
import torch
from net_class import Net
import warnings
import tkinter as my_tk
from tkinter import *
from tkinter import filedialog, ttk
from pathlib import Path
from tkinter import PhotoImage
from PIL import Image, ImageTk

warnings.filterwarnings("ignore")

############################################

def open_file_dialog():
    file_path = filedialog.askopenfilename(title="Select a File", filetypes=[("JPG files", "*.jpg"), ("All files", "*.*")])
    path = Path(file_path)

    if file_path:
        selected_file_label.config(text="Selected File: "+path.name)
        selected_file_label.config(font=("Roman", 14))
        process_file(file_path)


def process_file(file_path):
    # Implement your file processing logic here
    # For demonstration, let's just display the contents of the selected file

    # 50x50 pixels
    img_size = 50

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (img_size,img_size))

    img_array = np.array(img)

    img_array = img_array / 255

    img_array = torch.Tensor(img_array)

    net = Net()
    net.load_state_dict(torch.load("/Volumes/RGHardDrive/saved_model.pth"))
    net.eval()


    net_out = net(img_array.view(-1, 1, img_size,img_size))[0]
    


    if net_out[0] >= net_out[1]:
        #print("Prediction: BENIGN"+"\n"+"Confidence: "+str(net_out[0]))
        Result_file_label.config(text="Prediction: BENIGN\nConfidence: "+str(round(round(float(net_out[0]),3)*100,2))+"%")
        Result_file_label.config(font=("Roman", 15))
    else:

        Result_file_label.config(text="Prediction: MALIGNANT\nConfidence:"+str(round(round(float(net_out[1]),3)*100,2))+"%")
        Result_file_label.config(font=("Roman", 15))

#############################################



root=my_tk.Tk()
root.title("SCAN Application - Raghav Garg")
root.geometry("1800x1200")

bg = PhotoImage(file = "/Volumes/RGHardDrive/newbgimage.png")

label1 = Label(root, image = bg) 
label1.place(x = -140, y = 0) 

header_frame = my_tk.Frame(root, bg='#337CAB', bd=5)
header_frame.place(relx=0.5, rely=0.01, relwidth=0.7, relheight=0.11, anchor='n')
header_label = my_tk.Label(header_frame)
header_label.place(relwidth=1, relheight=1)
header_label['text']= '  ---------------------------------------------------------- \n \n SCAN - Skin Cancer Analysis Network \n ---------------------------------------------------------- \n Melenoma Skin Cancer Identifier using CNN (Convolutional Neural Networks). \n by Raghav Garg \n \n ----------------------------------------------------------'
header_label.config(font=("Arial Rounded MT Bold", 14))

header_frame = my_tk.Frame(root, bg='#337CAB', bd=5)
header_frame.place(relx=0.25, rely=0.15, relwidth=0.2, relheight=0.3, anchor='n')
header_label = my_tk.Label(header_frame)
header_label.place(relwidth=1, relheight=1)



open_button = my_tk.Button(root, text="Select an Image to SCAN", command=open_file_dialog)
open_button.pack(padx=10, pady=120)
open_button.place(x=270, y=135)
open_button.config(font=("Arial Rounded MT Bold", 12))


selected_file_label = my_tk.Label(root, text="Selected File:")
selected_file_label.pack()
selected_file_label.place(x=232, y=250)
selected_file_label.config(font=("Roman", 14))


Result_file_label = my_tk.Label(root, text="Prediction:\nConfidence:")
Result_file_label.place(x=232, y=300)
Result_file_label.config(font=("Roman", 15))

helping_label = my_tk.Label(root, text="Click Above to Upload a File \n and Test the Model")
helping_label.place(x=245, y=180)
helping_label.config(font=("Times New Roman", 18))



root.mainloop()