import numpy as np
import time
from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
import sys
import os
import imutils

'''
TODO: Better ROI extraction
      Get best thresholds (low and high) 
'''
def rgb_show(_colors):
  '''
  Helper method, represent true colors from RGB values
  '''
  stack = list()
  for num, color in enumerate(_colors):
    img = np.zeros((100,200,3), np.uint8)
    b, g, r = color[0], color[1], color[2]
    img[:] = [b,g,r]
    stack.append(img)

  cv2.imshow('Boje', np.hstack(stack))

def load_images(_background):
  '''
  Self explainable
  '''
  background = cv2.imread(_background)
  # connector = cv2.imread(_connector)
  # query = cv2.imread(_query)

  return background #, connector #, query

def low_level_processing(_background, _connector):
  '''
  Frame preparation for better color isolation
  '''
  
  gray_background = cv2.cvtColor(_background, cv2.COLOR_BGR2GRAY)
  gray_connector = cv2.cvtColor(_connector, cv2.COLOR_BGR2GRAY)
  
  delta = cv2.absdiff(gray_background, cv2.convertScaleAbs(gray_connector))
  prag = cv2.threshold(delta, 30, 255, cv2.THRESH_BINARY)[1]
  cv2.imwrite('prag.png', prag)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()
  # nadji nesto pametnije za ovo
  roi = _connector[440:440+20, 360:360+120]
  roi_t = prag[440:440+20, 360:360+120]
  
  return roi, roi_t

def get_colors(_roi, _roi_t):
  '''
  Isolates colors from ROIs
  for each wire returns average RGB value
  '''
  c_roi = _roi.copy()
  c_roi_t = _roi_t.copy()

  # _roi = cv2.cvtColor(_roi, cv2.COLOR_BGR2HSV)
  bounding_rects = list()
  (cnts, _) = cv2.findContours(c_roi_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  for_cnts = _roi.copy()
  if len(cnts) < 4:
    return []
  else:
    for c in cnts:
      x,y,w,h = cv2.boundingRect(c)
      bounding_rects.append(cv2.boundingRect(c))
      cv2.rectangle(for_cnts, (x, y), (x + w, y + h), (0, 255, 0), 1)

    bounding_rects = sorted(bounding_rects, key=lambda x: x[0])
    # print "pravokutnici: ", bounding_rects
    # if len(cnts) >= 4:
    cv2.imwrite('test.png', for_cnts)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    candidates = [[_roi[i][j] for i in xrange(b[1], b[1] + b[3]) for j in xrange(b[0], b[0] + b[2]) if _roi_t[i][j] == 255]
                 for b in bounding_rects]

    boje = list()
    for can in candidates:
      b, g, r = 0, 0, 0
      for c in can:
        b += c[0]
        g += c[1]
        r += c[2]
      b_avg = b / len(can)
      g_avg = g / len(can)
      r_avg = r / len(can)
      boje.append([b_avg, g_avg, r_avg])

    return boje

def compare(_q_colors, _referent_colors):
  '''
  Compares query colors with referent colors
  Returns True only if all colors presented in referent model
  are matched
  '''
  Thlb, Thlg, Thlr = 25, 25, 25
  Thhb, Thhg, Thhr = 50, 50, 50
  same = list()
  for q, r in zip(_q_colors, _referent_colors):
    print "Query color", q
    print "Referent color", r
    if abs(q[0] - r[0]) < Thlb \
      and abs(q[1] - r[1]) < Thlg \
      and abs(q[2] - r[2]) < Thlr:
      # print "Query color: ", q 
      # print "Referent color: ", r
      print "Colors are approximately same!"
      same.append(True)
    elif abs(q[0] - r[0]) > Thhb \
      and abs(q[1] - r[1]) > Thhg \
      and abs(q[2] - r[2]) > Thhr:
      # print "Query color", q
      # print "Referent color", r 
      print "Colors are different!"
      same.append(False)

  print same
  if all(same) and len(same) == len(_referent_colors):
    return True
  elif all(same) and len(same) != len(_referent_colors):
    print "Nema dovoljno boja!\nBolje podesite konektor"
    return False
  else:
    return False

def get_thresholds(_referent_colors):
  '''
  Calculates referent colors thresholds
  Takes referent colors list and as an output gives average
  RGB channel for each wire
  '''
  conn = zip(*_referent_colors)
  thresh = list()
  for c in conn:
      # print "Color: ", c
    thresh.append(map(sum, zip(*c)))
  average_thresh = list()
  for t in thresh:
    average_thresh.append(map(lambda x: x / len(conn[0]), t))

  return average_thresh

def main():
  '''
  >>>usage #1: python color_det.py <background_image>
  >>>usage #2: ./color_det.py <background_image>
  >>>for background image capture use take_picture.py script

  REFERENT MODEL:
    Loads input images (background and background+connector)
    Finds binarized image of connector and extracts ROI
    Color inspection in ROI
  
  QUERY:
    Loads new connector image
    Comparison against referent model
    Match?
  '''
  if len(sys.argv) < 2:
    sys.stderr.write("Potrebno 2 argumenata (path to image)")
    sys.exit(1)

  camera = PiCamera()
  camera.resolution = (800, 600)

  capture = PiRGBArray(camera, size=(800, 600))
  time.sleep(2)
  # query = None
  background = load_images(sys.argv[1])
  # referent_colors = [(2, 16, 18), (17, 43, 10), (0, 15, 102), (64, 31, 7)]
  # connector = cv2.imread(raw_connector)
  # roi, roi_t = low_level_processing(background, raw_connector)
  # referent_colors = get_colors(roi, roi_t)
  cnt = 0
  referent = list()
  for f in camera.capture_continuous(capture, format="bgr", use_video_port=True):
    # print "STREAM RADI"
    cnt += 1
    print "Trenutni frame broj: ", cnt
    frame = f.array
    # cv2.imwrite('frame.png', frame)
    if cnt <= 4:
      # s = "Referentni konektor"
      roi, roi_t = low_level_processing(background, frame)
      referent.append(get_colors(roi, roi_t))
      # print "Referentne boje: ",  referent
      # cv2.imshow('VideoStream', frame)
      # cv2.putText(frame, s, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
      capture.truncate(0)
    elif cnt == 5:
      # s = "Referentne boje"
      thresholds = get_thresholds(referent)
      print "Pragovi za boje: ", thresholds
      time.sleep(7) # stoj dok se ne stavi query connector
      # cv2.putText(frame, s, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
      # cv2.imshow('VideoStream', frame)
      # key = cv2.waitKey(1) & 0xFF
      # if key == ord("q"):
        # break
      capture.truncate(0)
    # print "Frame: ", frame.shape
    # print "Background: ", background.shape
    else:
      q_roi, q_roi_t = low_level_processing(background, frame)
      q_colors = get_colors(q_roi, q_roi_t)
      # rgb_show(q_colors)
      if len(q_colors) >= 4:
        # s = "Pronadjen konektor"
        are_same = compare(q_colors, thresholds)
        if are_same:
          s = "Wire colors are same!"
          cv2.putText(frame, s, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
          # os.system('mpg321  ~/cable_color/clap.mp3')
        else:
          s = "Wire colors are different, repair that!"
          cv2.putText(frame, s, (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
          os.system('mpg321  ~/cable_color/beep-02.mp3')
        
        cv2.imshow('VideoStream', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
          break
        # cnt += 1
        capture.truncate(0)
        # time.sleep(5)
      else:
        cv2.imshow('VideoStream', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
          break
        # cnt += 1
        capture.truncate(0)
        continue

      capture.truncate(0)
    # cnt += 1

if __name__ == '__main__':
  main()