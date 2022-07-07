from shutil import register_unpack_format
import cv2 as cv
import numpy as np
import pyautogui
import time
from skimage import measure
from imutils import contours
import imutils
import pandas as pd
import playsound
import os
from random import choice
from win32gui import GetWindowText, GetForegroundWindow
from random import randint

def filter_color(img, color, for_display=False):
    copy = img.copy()
    if for_display:
        if color == 'blue':
            copy[:, :, 0] = 0
            copy[:, :, 1] = 0
        elif color == 'green':
            copy[:, :, 0] = 0
            copy[:, :, 2] = 0
        elif color == 'red':
            copy[:, :, 1] = 0
            copy[:, :, 2] = 0
    else:
        if color == 'blue':
            copy = copy[:, :, 0]
        elif color == 'green':
            copy = copy[:, :, 1]
        elif color == 'red':
            copy = copy[:, :, 2]
    return copy    

def to_grayscale(img):
    copy = img.copy()
    copy = cv.cvtColor(copy, cv.COLOR_BGR2GRAY)
    return copy

def display(img):
    cv.imshow('', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def filter_radar(img):
    radar = img.copy()
    radar = radar[849:1041,60:279]
    # set the middle of the image to black with a 5 pixel radius
    center = (radar.shape[1]//2+1, radar.shape[0]//2-2)
    cv.circle(radar, center, 9, (0, 0, 0), -1)
    return radar, center

def find_contours(img):
	labels = measure.label(img, background=0)
	mask = np.zeros(img.shape, dtype="uint8")
	# loop over the unique components
	for label in np.unique(labels):
		# if this is the background label, ignore it
		if label == 0:
			continue
		# otherwise, construct the label mask and count the
		# number of pixels 
		labelMask = np.zeros(img.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv.countNonZero(labelMask)
		# if the number of pixels in the component is sufficiently
		# large, then add it to our mask of "large blobs"
		if numPixels > 15:
			mask = cv.add(mask, labelMask)
	return mask

def draw_blobs(img, mask, print_image=False):
	cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL,
		cv.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	try:
		cnts = contours.sort_contours(cnts)[0]
	except:
		return [], 0
	blob_centers = []
	final_radar = np.array(img)

	for (i, c) in enumerate(cnts):
		(x, y), radius = cv.minEnclosingCircle(c)
		blob_center = (int(x), int(y))
		blob_centers.append(blob_center)
		radius = int(radius)
		cv.circle(final_radar, blob_center, radius, (0, 0, 0), 1)
		cv.putText(final_radar, "{}".format(i + 1), tuple([int(round(x+4)), int(round(y-2)) ]),
			cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
	if print_image:
		display(final_radar)
	return blob_centers, len(cnts)

def compute_angle(center, blob):
    x, y = center
    x1, y1 = blob
    angle = np.arctan2(y1-y, x1-x)
    # convert in degrees
    angle = np.degrees(angle)
    return angle+180

def get_quadrant(angle):
    if angle > 45 and angle <= 135:
        return 'front'
    elif angle > 135 and angle <= 225:
        return 'right'
    elif angle > 225 and angle <= 315:
        return 'behind'
    else:
        return 'left'

def scrap_radar(img, allies=False, debug=False):
    radar, center = filter_radar(img)
    if allies:
        radar = filter_color(radar, "blue")
    else:
        radar = filter_color(radar, "red")
    
    blurred = cv.GaussianBlur(radar, (9, 9), 0)
    thresh = cv.threshold(blurred, 230, 255, cv.THRESH_BINARY)[1]
    mask = find_contours(thresh)
    blob_centers, num_blobs = draw_blobs(radar, mask, print_image=debug)
    
    if num_blobs == 0:
        if debug:
            print(f'Nobody detected')
        return []
    else:
        if debug:
            for i in range(num_blobs):
                if allies:
                    print(f"Friendly at position: {get_quadrant(compute_angle(center, blob_centers[i]))}")
                else:
                    print(f"Enemy at position: {get_quadrant(compute_angle(center, blob_centers[i]))}")
        return [get_quadrant(compute_angle(center, blob_centers[i])) for i in range(num_blobs)]

def count_quadrants(quadrants):
    counts = {'front': 0, 'left': 0, 'right': 0, 'behind': 0}
    for quadrant in quadrants:
        counts[quadrant] += 1
    return counts

def filter_shield_bar(img):
    copy = img.copy()
    copy = copy[115,755:1165].reshape(1,410,3)
    # copy = copy[110:125,755:1165]
    return copy

def filter_health_bar(img):
    copy = img.copy()
    copy = copy[127,835:1085].reshape(1,250,3)
    return copy

def get_bar_status(img, shield=True):
    if shield:
        img = filter_shield_bar(img)
    else :
        img = filter_health_bar(img)
    img = filter_color(img, "blue")
    # get percentage of bar filled
    if shield:
        img = cv.threshold(img, 128, 255, cv.THRESH_BINARY)[1]
    else:
        img = cv.threshold(img, 60, 255, cv.THRESH_BINARY)[1]
    percentage = ( cv.countNonZero(img) / img.shape[1] ) * 100
    return percentage

def filter_kill_feed(img):
    copy = img.copy()
    copy = copy[540:640,50:70]
    return copy

def scrap_kill_feed(img, debug=False):
    kill_feed = filter_kill_feed(img)
    kill_feed = to_grayscale(kill_feed)
    thresh = cv.threshold(kill_feed, 250, 255, cv.THRESH_BINARY)[1]
    mask = find_contours(thresh)
    _, num_blobs = draw_blobs(kill_feed, mask, print_image=debug)
    return num_blobs

def filter_death(img):
    copy = img.copy()
    copy = copy[835:870,75:110] # gets gm logo on blue background
    # copy = copy[1050:1065,1755:1800] # gets gm logo on red background
    return copy

def detect_death(img, debug=False):
    img = filter_color(img, "blue")
    img = filter_death(img)
    avg = get_average_pixel_value(img)
    if debug:
        print(f"death dexpected value: [234;237], and got : {avg}")
    return 234 < avg < 237

def filter_transition(img, season=2):
    copy = img.copy()
    if season == 1:
        copy = copy[980:1080,1825:1915]
    else:
        copy = copy[1072:1080,0:1920]
    return copy

def filter_menu(img):
    copy = img.copy()
    # copy = copy[100:320,100:260] # gets gm logo on blue background
    copy = copy[1051:1065,1770:1790]
    return copy

def filter_pause(img):
    copy = img.copy()
    copy = copy[100:320,100:260] # gets gm logo on blue background
    # copy = copy[1050:1065,1755:1800] # gets gm logo on red background
    return copy

def get_average_pixel_value(img):
    img = img.reshape(1, -1)
    img = img.astype(np.float32)
    return np.average(img)

def filter_tab(img):
    copy = img.copy()
    copy = copy[550:610,100:165] # gets gm logo on blue background
    # copy = copy[1050:1065,1755:1800] # gets gm logo on red background
    return copy

def detect_tab(img, debug=False):
    img = filter_color(img, "blue")
    img = filter_tab(img)
    avg = get_average_pixel_value(img)
    if debug:
        print(f"tab dexpected value: [183;186], and got : {avg}")
    return 183 < avg < 186

def detect_transition(img, debug=False):
    img = filter_transition(img)
    avg = get_average_pixel_value(img)
    if debug:
        print(f"transition dexpected value: [2;3], and got : {avg}")
    return 2 < avg < 3

def detect_pause(img, debug=False):
    img = filter_color(img, "blue")
    img = filter_pause(img)
    avg = get_average_pixel_value(img)
    if debug:
        print(f"pause dexpected value: [140;145], and got : {avg}")
    return 140 < avg < 145

def detect_menu(img, debug=False):
    img = filter_menu(img)
    avg = get_average_pixel_value(img)
    if debug:
        print(f"menu dexpected value: [207;212], and got : {avg}")
    return 207 < avg < 213

def filter_main_weapon(img):
    copy = img.copy()
    copy = copy[950:1010,1700:1825] # gets gm logo on blue background
    return copy

def filter_secondary_weapon(img):
    copy = img.copy()
    copy = copy[895:925,1580:1660] # gets gm logo on red background
    return copy

def process_weapons(img, main=True):
    if main:
        weapon = filter_main_weapon(img)
    else:
        weapon = filter_secondary_weapon(img)
    blue = filter_color(weapon, "blue")
    binary = cv.threshold(blue, 200, 255, cv.THRESH_TOZERO)[1]
    # binary[binary == 255] = 0
    return binary

def get_distance(img, weapon):
    img = img.reshape(1, -1)
    img = img.astype(np.float32)
    weapon = weapon.reshape(1, -1)
    weapon = weapon.astype(np.float32)
    return np.linalg.norm(img - weapon)

def retrieve_weapons(main=True):
    weapon_list = {}
    if main:
        path = './weapons/main/'
    else:
        path = './weapons/secondary/'
    for filename in os.listdir(path):
        img = cv.imread(path + filename, 0)
        weapon_list[filename.replace(".png", "")] = img
    return weapon_list

def recognize_weapon(img, main=True, debug=False):
    value = 10000
    filtered_img = process_weapons(img, main=main)
    if main:
        weapon_list = main_weapons
    else:
        weapon_list = secondary_weapons
    for name, weapon in weapon_list.items():
        distance = get_distance(filtered_img, weapon)
        if debug:
            print(f"Similarity with {name}: {distance}")
        if distance < value:
            value = distance
            best_weapon = name
    if value > 2500:
        return 'unknown'
    else:
        if debug:
            Verti = np.concatenate((filtered_img, weapon_list[best_weapon]), axis=0)
            cv.imshow('VERTICAL', Verti)
            cv.waitKey(0)
        return best_weapon

def record_state(df, frame, i, debug=False):
    allies = count_quadrants(scrap_radar(frame, allies=True, debug=debug))
    for key, value in allies.items():
        df.loc[i, f"ally_{key}"] = value
    enemies = count_quadrants(scrap_radar(frame, allies=False, debug=debug))
    for key, value in enemies.items():
        df.loc[i, f"enemy_{key}"] = value
    df.loc[i, 'shield'] = get_bar_status(frame, shield=True)
    df.loc[i, 'health'] = get_bar_status(frame, shield=False)
    df.loc[i, 'kill_feed'] = scrap_kill_feed(frame, debug=debug)
    df.loc[i, 'main_weapon'] = recognize_weapon(frame, main=True, debug=debug)
    df.loc[i, 'secondary_weapon'] = recognize_weapon(frame, main=False, debug=debug)
    return df

def call_cortana_for_help(df, i, last_callout, img, debug=False, save_images=True):
    time_elapsed = time.time() - last_callout
    if (randint(0, int(time_elapsed)) < 10): # (time_elapsed < 15) and 
        # if debug:
        #     print("You've done enough Cortana chill out a bit")
        return last_callout
    else:
        path = "C:\\Users\\valen\\Music\\weapon\\"

        if df.loc[i, 'enemy_behind'] > 0 and randint(0, 100) < 10:
            mp3 = choice(os.listdir(path + "behind you\\"))
            if debug:
                print(f"enemies behind you, playing {mp3}")
            if save_images:
                img.save(f"{i}.png")
            playsound.playsound(path + "behind you\\" + mp3, block=False)

        else:

            if df.loc[i, 'kill_feed'] > 0 and randint(0, 100) < 100:
                mp3 = choice(os.listdir(path + "target killed\\"))
                if debug:
                    print(f"kill feed detected, playing {mp3}")
                if save_images:
                    img.save(f"{i}.png")
                playsound.playsound(path + "target killed\\" + mp3, block=False)
                
            else:

                if (df.loc[i, 'main_weapon'] != 'unknown') and (df.loc[i, 'main_weapon'] != df.loc[i-1, 'main_weapon']) and (df.loc[i, 'main_weapon'] != df.loc[i-1, 'secondary_weapon']) and randint(0, 100) < 75:
                    mp3 = choice(os.listdir(path + "nice find\\"))
                    if debug:
                        print(f"weapon change detected, playing {mp3}")
                    if save_images:
                        img.save(f"{i}.png")
                    playsound.playsound(path + "nice find\\" + mp3, block=False)
                    
                else:

                    if (df.loc[i, 'enemy_front'] > 0 or df.loc[i, 'enemy_left'] > 0 or df.loc[i, 'enemy_right'] > 0) and randint(0, 100) < 75:
                        mp3 = choice(os.listdir(path + "target to kill\\"))
                        if debug:
                            print(f"enemies in front, playing {mp3}")
                        if save_images:
                            img.save(f"{i}.png")
                        playsound.playsound(path + "target to kill\\" + mp3, block=False)
                        
                    else:

                        if df.loc[i, 'shield'] < df.loc[i-1, 'shield'] and randint(0, 100) < 75:
                            mp3 = choice(os.listdir(path + "lookout\\"))
                            if debug:
                                print(f"shield damaged, playing {mp3}")
                            if save_images:
                                img.save(f"{i}.png")
                            playsound.playsound(path + "lookout\\" + mp3, block=False)
                            
                        else:

                            if df.loc[i, 'shield'] > df.loc[i-1, 'shield'] and randint(0, 100) < 75:
                                mp3 = choice(os.listdir(path + "clear\\"))
                                if debug:
                                    print(f"shield restored, playing {mp3}")
                                if save_images:
                                    img.save(f"{i}.png")
                                playsound.playsound(path + "clear\\" + mp3, block=False)

                            else:

                                if randint(0,100) < 50:
                                    mp3 = choice(os.listdir(path + "quips\\"))
                                    if debug:
                                        print(f"RAS {mp3}")
                                    playsound.playsound(path + "quips\\" + mp3, block=False)
                                    
        last_callout = time.time()
        return last_callout

def cortana(df, decision=True, debug=False, skip=False, save_images=False):
    playsound.playsound("C:\\Users\\valen\\Music\\weapon\\hello.mp3", block=False)
    prev = 0
    i = 1
    last_callout = time.time()

    while True:
        time_elapsed = time.time()-prev
        if time_elapsed > 1:
            img = pyautogui.screenshot()
            # if save_images and i%10 == 0:
            #     img.save(f"{i}.png")
            prev = time.time()
            frame = img.copy()
            frame = np.array(frame)
            frame = frame[:, :, ::-1].copy()
            if GetWindowText(GetForegroundWindow()) != "Halo Infinite":
                if skip:
                    print("Halo is not in focus")
                continue
            elif stfu_cortana(frame, debug=False, skip=skip):
                last_callout +=1
                continue
            else:
                df = record_state(df, frame, i, debug)
                last_callout = call_cortana_for_help(df, i, last_callout, img, decision, save_images)
                i+=1
        # cv.waitKey(1000)

def stfu_cortana(frame, debug=False, skip=False):
    if detect_menu(frame, debug):
        if skip:
            print(f"Stfu Cortana im in menu")
        return True
    elif detect_transition(frame, debug):
        if skip:
            print(f"Stfu Cortana im in transition")
        return True
    elif detect_pause(frame, debug):
        if skip:
            print(f"Stfu Cortana im in pause")
        return True
    elif detect_death(frame, debug):
        if skip:
            print(f"Stfu Cortana im dead")
        return True
    elif detect_tab(frame, debug):
        if skip:
            print(f"Stfu Cortana im visualization how good i am")
        return True
    else:
        return False

main_weapons = retrieve_weapons(main=True)
secondary_weapons = retrieve_weapons(main=False)
