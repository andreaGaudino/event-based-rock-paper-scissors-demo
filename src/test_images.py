import time
import cv2
import numpy as np
import glob
import os
from utils import classify_img, majority_vote, IMSIZE, PRED_TO_SYMBOL, WINNING_MOVES




def run_offline_mode(camera_name, screen, interpreter, input_details, output_details, 
                     voter, winning_imgs, SCREEN_W, SCREEN_H):
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_paths = sorted(glob.glob(os.path.join(base_dir, "..", "assets", "sample_frames", "*.png")))
    print(base_dir)
    if not image_paths:
        print("Error: image not found in sample_frames")
        return

    print(f"Found {len(image_paths)} images for testing. Starting...")
    
    running = True

    prediction_time = []

    # --- GAME LOOP ---
    for img_path in image_paths:
        if not running:
            break
        start = time.time()
        # Reading image with OpenCV (format needed by the NN)
        frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue
        #screen.fill(0)
        screen = np.zeros((int(SCREEN_H), int(SCREEN_W), 3), dtype=np.uint8)

        img_resized = cv2.resize(frame, (IMSIZE, IMSIZE), interpolation=cv2.INTER_AREA)
        
        # Prediction phase
        
        pred_name, pred_idx, pred_vector = classify_img(img_resized, interpreter, input_details, output_details)
        final_vote_idx = voter.new_prediction_and_vote(pred_idx)
        
        
        # We display the flow of images on the left 
        img_size = int(SCREEN_H*0.6)
        cam_view = cv2.resize(img_resized, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        cam_view_color = cv2.cvtColor(cam_view, cv2.COLOR_GRAY2RGB)
        

        img_x = int(SCREEN_W/20)
        img_y = int(SCREEN_H/15)
        # Place cam view into screen (rows=y, cols=x)
        screen[img_y:img_y + img_size, img_x: img_x + img_size] = cam_view_color
        
        # Writing the name of the file beneath the img
        txt_file = f"File: {os.path.basename(img_path)}"
        cv2.putText(screen, txt_file, (img_x, img_y + img_size + img_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

        if final_vote_idx is not None:
            confirmed_move = PRED_TO_SYMBOL[final_vote_idx]
            winning_move = WINNING_MOVES[confirmed_move]
            
            # Testing results
            txt_you = f"Your move: {confirmed_move.upper()}"
            cv2.putText(screen, txt_you, (img_x, img_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            txt_model = f"Model move: {winning_move.upper()}"
            winning_img_x = SCREEN_W - img_size - int(SCREEN_W/20)
            cv2.putText(screen, txt_model, (int(winning_img_x), img_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display the winning image (robust: handle alpha channel and out-of-bounds)
            if winning_move in winning_imgs:
                winning_surface = winning_imgs[winning_move]
                winning_img_resized = cv2.resize(winning_surface, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
                y1 = int(img_y)
                y2 = y1 + winning_img_resized.shape[0]
                x1 = int(winning_img_x)
                x2 = x1 + winning_img_resized.shape[1]
                # If image has alpha channel, convert to RGB
                if winning_img_resized.ndim == 3 and winning_img_resized.shape[2] == 4:
                    b, g, r, a = cv2.split(winning_img_resized)
                    alpha = (a.astype(float) / 255.0)[..., None]
                    src_rgb = cv2.merge([b, g, r]).astype(float)
                    dst_rgb = screen[y1:y2, x1:x2].astype(float)
                    comp = src_rgb * alpha + dst_rgb * (1.0 - alpha)
                    screen[y1:y2, x1:x2] = comp.astype(np.uint8)
                else:
                    # No alpha: attempt a simple chroma-key to ignore uniform background
                    win = winning_img_resized
                    if win.ndim == 2:
                        win = cv2.cvtColor(win, cv2.COLOR_GRAY2BGR)
                    # take top-left pixel as background sample
                    bg = win[0, 0].astype(int)
                    diff = np.linalg.norm(win.astype(int) - bg, axis=2)
                    mask = diff > 30  # threshold; tweak if needed
                    # copy only pixels where mask is True
                    dst = screen[y1:y2, x1:x2]
                    for c in range(3):
                        ch = dst[..., c]
                        ch[mask] = win[..., c][mask]
                        dst[..., c] = ch
                    screen[y1:y2, x1:x2] = dst

            # Probabilities
            txt_prob = "Outcome probabilities:"
            (text_width, text_height), _ = cv2.getTextSize(PRED_TO_SYMBOL[3].upper(), cv2.FONT_HERSHEY_SIMPLEX, 1, 1)  
            cv2.putText(screen, txt_prob, (img_x, img_y + img_size + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            right = img_x + text_width 
            
            start_y = img_y + img_size + 80 +  text_height
            for i in range(len(pred_vector)):
                label = f"{PRED_TO_SYMBOL[i].upper()}"
                cv2.putText(screen, label, (img_x, start_y + i*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
                # bar lenght
                bar_w = int(400 * float(pred_vector[i]))
                color = (0, 200, 100) if bar_w > 15 else (50, 50, 200)
                cv2.rectangle(screen, (right + 20, start_y + i*40 - label_height), (right + 20 + bar_w, start_y + i*40 + 10), color, -1)

        
        cv2.imshow(camera_name, screen)
        end = time.time()

        prediction_time.append(end-start)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            running = False
        if cv2.getWindowProperty(camera_name, cv2.WND_PROP_VISIBLE) < 1:
            running = False
    print(f"Average prediction time: {np.average(prediction_time)*1000:.2f} ms")

        # clock.tick(20)    # 30FPS cicle (close to real world)