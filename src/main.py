import argparse
import cv2
import numpy as np
import time
import glob
import os
import tensorflow as tf
import sys
import ctypes
from datetime import datetime
from test_images import run_offline_mode
from utils import classify_img, majority_vote, IMSIZE, PRED_TO_SYMBOL, WINNING_MOVES, CAMERA_ON, capture
from read_camera import run_reading_camera_live




def parse_args():
    p = argparse.ArgumentParser(description="Run the Roshambo demo")
    #p.add_argument("--recording", type=bool, default=False, help='If the input is a recorded file, please type the --recording option')
    p.add_argument("--recording", "-r", action="store_true",
                   help='If set, record statistics to ./statistics')
    p.add_argument("--model", "-m", default = "finetuned", choices=["finetuned", "original"], 
                   help="Choose which model to use: finetuned or original")
    p.add_argument("--mode", "-o", default = "offline", choices=["offline", "live"],
                   help="Choose which mode to run: offline (test images) or live (camera input)")
    return p.parse_args()

# --- MAIN GAME LOGIC ---

def main():
    args = parse_args()
    recording = args.recording
    print("Model loading...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_name = args.model
    if model_name == 'finetuned':
        model_path = os.path.join(project_root, "model", "finetuned_model_dextra_roshambo_new.tflite")
    else:
        model_path = os.path.join(project_root, "model", "dextra_roshambo.tflite")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    voter = majority_vote(window_length=5, num_classes=4)

    if sys.platform.startswith('win'):
        try:
            user32 = ctypes.windll.user32
            SCREEN_W = user32.GetSystemMetrics(0)
            SCREEN_H = user32.GetSystemMetrics(1)*0.9
        except Exception:
            SCREEN_W, SCREEN_H = 1000, 1000
    else:
        try:
            import tkinter as tk
            root = tk.Tk()
            SCREEN_W = root.winfo_screenwidth()
            SCREEN_H = root.winfo_screenheight()
            root.destroy()
        except Exception:
            SCREEN_W, SCREEN_H = 1000, 800    
    
    cv2.namedWindow("DAVIS Live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("DAVIS Live", int(SCREEN_W), int(SCREEN_H))
    screen = np.zeros((int(SCREEN_H), int(SCREEN_W), 3), dtype=np.uint8)
    

    # Loading winning moves images
    print("Loading winning images...")
    winning_imgs = {}
    for move in ['rock', 'paper', 'scissors']:
        img_path = os.path.join(project_root, 'assets', 'symbols', f'{move}.png')
        if os.path.exists(img_path):
            winning_imgs[move] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)


    f_csv = None
    f_txt = None
    if recording:
        now = datetime.now()
        file_name = now.strftime('%Y_%m_%d_%H_%M_%S')

        stats_dir = os.path.join(project_root, 'statistics', file_name)
        os.makedirs(stats_dir, exist_ok=True)
        f_csv = open(os.path.join(stats_dir, f'{file_name}.csv'), 'a', newline='')
        f_txt = open(os.path.join(stats_dir, f'{file_name}.txt'), 'a')


    if args.mode == 'offline':
        run_offline_mode(screen=screen,
                     camera_name='DAVIS Live',
                     interpreter=interpreter,
                     input_details=input_details,
                     output_details=output_details,
                     voter=voter,
                     winning_imgs=winning_imgs,
                     SCREEN_W=SCREEN_W,
                     SCREEN_H=SCREEN_H)
    else:  
        run_reading_camera_live(capture=capture, 
                                camera_name='DAVIS Live',
                                screen=screen,
                                interpreter=interpreter,
                                input_details=input_details,
                                output_details=output_details,
                                voter=voter,
                                winning_imgs=winning_imgs,
                                SCREEN_W=SCREEN_W,
                                SCREEN_H=SCREEN_H,
                                csv_stats_file = f_csv,
                                txt_stats_file = f_txt)
    if recording:
        f_csv.close()
        f_txt.close()

        
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()