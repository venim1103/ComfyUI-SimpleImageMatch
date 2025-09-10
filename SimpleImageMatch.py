import numpy as np
import cv2 as cv
from PIL import Image
import os

R,G,B,H,S,V,Y,U,L,A = 0,1,2,0,1,2,0,1,0,1
LUMA_WEIGHTS = [0.299, 0.587, 0.114]
MAX8BITNUM = np.iinfo('uint8').max

def match_deviation_difference(Ch_ref, Ch_target):
    # Match deviation
    std_ref = float(np.std(Ch_ref))
    std_target = float(np.std(Ch_target))
    Ch_target_deviation = Ch_target * (std_ref / std_target) if std_target > 0 else Ch_target
    # Match difference
    mean_ref = float(np.mean(Ch_ref))
    mean_target_deviation = float(np.mean(Ch_target_deviation))
    Ch_target_out = Ch_target_deviation + (mean_ref - mean_target_deviation)
    return Ch_target_out

def match_images(img_a, img_b):
    # Turn into LAB
    lab_a = cv.cvtColor(np.array(img_a), cv.COLOR_RGB2LAB)
    lab_b = cv.cvtColor(np.array(img_b), cv.COLOR_RGB2LAB)
    Al, Aa, Ab = lab_a[:,:,L], lab_a[:,:,A], lab_b[:,:,B]
    Bl, Ba, Bb = lab_b[:,:,L], lab_b[:,:,A], lab_b[:,:,B]
    # Do statistical matching for contrast, brightness, color and saturation
    Bl_out = match_deviation_difference(Al, Bl)
    Ba_out = match_deviation_difference(Aa, Ba)
    Bb_out = match_deviation_difference(Ab, Bb)
    # Combine back to lab_b_out
    lab_b_out = cv.merge([Bl_out, Ba_out, Bb_out])
    # Convert back to 8 bit
    lab_b_out = np.clip(lab_b_out, 0, MAX8BITNUM).astype('uint8')
    # Turn back into RGB
    img_b_out = cv.cvtColor(lab_b_out, cv.COLOR_LAB2RGB)
    return img_b_out

def get_grayscale(img):
    img_array = np.array(img).astype('float') / MAX8BITNUM
    luminance = LUMA_WEIGHTS[R] * img_array[:,:,R] + LUMA_WEIGHTS[G] * img_array[:,:,G] + LUMA_WEIGHTS[B] * img_array[:,:,B]
    grayscale = np.uint8(luminance * MAX8BITNUM)
    return grayscale

if __name__ == "__main__":
    # Example usage
    img_a_path = os.sys.argv[1] if len(os.sys.argv) > 1 else "a.jpg"
    img_b_path = os.sys.argv[2] if len(os.sys.argv) > 2 else "b.jpg"
    output_path = os.sys.argv[3] if len(os.sys.argv) > 3 else "out.jpg"
    grayscale_path = os.sys.argv[4] if len(os.sys.argv) > 4 else "grayscale.jpg"

    if os.path.exists(img_a_path) and os.path.exists(img_b_path):
        img_a = Image.open(img_a_path).convert("RGB")
        img_b = Image.open(img_b_path).convert("RGB")

        output_img = match_images(img_a, img_b)
        Image.fromarray(output_img).save(output_path)
        print(f"Matched image saved to {output_path}")

        grayscale_img = get_grayscale(img_a)
        Image.fromarray(grayscale_img).save(grayscale_path)
        print(f"Grayscale image saved to {grayscale_path}")
    else:
        print("Please ensure both image paths are correct.")
