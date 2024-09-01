from pathlib import Path
import cv2
import depthai
import numpy as np

USE_BLOBCONVERTER = True

USE_TRACKING = True
#TRACKER_TYPE = "Short Term KCF" # best? but only 60 objects, very slow, box lags behind
#TRACKER_TYPE = "Short Term Imageless" # up to 1000 objects, slow, box lags behind
TRACKER_TYPE = "Zero Term" # no history, up to 1000 objects, tracks the fastest

# NOTE:
# Person detection could be run on the camera on the unobstructed frame.
# However, this is a test to see if common neural networks are able to detect
# persons with blacked out face without having to retrain them.
TRY_PERSON_DETECTION = True
PERSON_DETECTION_NN = "pedestrian-detection-adas-0002"
#PERSON_DETECTION_NN = "person-detection-retail-0013"

if USE_BLOBCONVERTER:
    import blobconverter

pipeline = depthai.Pipeline()

cam_rgb = pipeline.createColorCamera()
cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setPreviewSize(1080, 1080)
cam_rgb.setInterleaved(False)
cam_rgb.setVideoSize(1080, 1080)

scale_nn = pipeline.createImageManip()
scale_nn.initialConfig.setResize(300, 300)
cam_rgb.preview.link(scale_nn.inputImage)

detection_nn = pipeline.createMobileNetDetectionNetwork()
if USE_BLOBCONVERTER:
    detection_nn.setBlobPath(blobconverter.from_zoo(
        name="face-detection-retail-0044",
        shaves=6,
        ))
else:
    detection_nn.setBlobPath(str((Path(__file__).parent / Path('face-detection-retail-0044_openvino_2021.4_6shave.blob')).resolve().absolute()))
detection_nn.setConfidenceThreshold(0.3)
scale_nn.out.link(detection_nn.input)

if USE_TRACKING:
    tracker = pipeline.createObjectTracker()
    tracker.setDetectionLabelsToTrack([1])
    if TRACKER_TYPE == "Short Term KCF":
        tracker.setTrackerType(depthai.TrackerType.SHORT_TERM_KCF)
    if TRACKER_TYPE == "Short Term Imageless":
        tracker.setTrackerType(depthai.TrackerType.SHORT_TERM_IMAGELESS)
    if TRACKER_TYPE == "Zero Term":
        tracker.setTrackerType(depthai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    tracker.setTrackerIdAssignmentPolicy(depthai.TrackerIdAssignmentPolicy.SMALLEST_ID)
    tracker.setTrackerThreshold(0.5) # detect early, track with confidence
    detection_nn.passthrough.link(tracker.inputDetectionFrame)
    detection_nn.passthrough.link(tracker.inputTrackerFrame)
    detection_nn.out.link(tracker.inputDetections)

sc_block = pipeline.create(depthai.node.Script)
sc_block.inputs['nn_in'].setBlocking(True)
sc_block.inputs['nn_in'].setQueueSize(1)
sc_block.inputs['frame_in'].setBlocking(True)
sc_block.inputs['frame_in'].setQueueSize(1)
if USE_TRACKING:
    tracker.out.link(sc_block.inputs['nn_in'])
else:
    detection_nn.out.link(sc_block.inputs['nn_in'])
cam_rgb.preview.link(sc_block.inputs['frame_in'])
if USE_TRACKING:
    sc_block.setScript("""
while True:
    frame = node.io['frame_in'].get()
    tracklets = node.io['nn_in'].get().tracklets

    framedata = frame.getData()

    width = frame.getWidth()
    height = frame.getHeight()
    framelength = width * height
    scale = 1080

    for tracklet in tracklets:
        upperleftx = int(tracklet.roi.x * scale)
        upperlefty = int(tracklet.roi.y * scale)
        lowerrightx = int((tracklet.roi.x + tracklet.roi.width) * scale)
        lowerrighty = int((tracklet.roi.y + tracklet.roi.height) * scale)

        boxwidth = lowerrightx-upperleftx

        for yvalue in range(upperlefty, lowerrighty):
            framedata[yvalue*width+upperleftx:yvalue*width+lowerrightx] = bytearray(boxwidth)
            framedata[framelength+yvalue*width+upperleftx:framelength+yvalue*width+lowerrightx] = bytearray(boxwidth)
            framedata[framelength+framelength+yvalue*width+upperleftx:framelength+framelength+yvalue*width+lowerrightx] = bytearray(boxwidth)
    
    node.io['host'].send(frame)
""")
else:
    # it seems there is a problem with the width of the bounding box
    sc_block.setScript("""
while True:
    frame = node.io['frame_in'].get()
    detections = node.io['nn_in'].get().detections

    framedata = frame.getData()

    width = frame.getWidth()
    height = frame.getHeight()
    framelength = width * height
    scale = 1080

    for detection in detections:
        upperleftx = int(detection.xmin * scale)
        upperlefty = int(detection.ymin * scale)
        lowerrightx = int(detection.xmax * scale)
        lowerrighty = int(detection.ymax * scale)

        boxwidth = lowerrightx-upperleftx

        for yvalue in range(upperlefty, lowerrighty):
            framedata[yvalue*width+upperleftx:yvalue*width+lowerrightx] = bytearray(boxwidth)
            framedata[framelength+yvalue*width+upperleftx:framelength+yvalue*width+lowerrightx] = bytearray(boxwidth)
            framedata[framelength+framelength+yvalue*width+upperleftx:framelength+framelength+yvalue*width+lowerrightx] = bytearray(boxwidth)
    
    node.io['host'].send(frame)
""")

xout_script = pipeline.createXLinkOut()
xout_script.setStreamName("host")
sc_block.outputs['host'].link(xout_script.input)

if TRY_PERSON_DETECTION:
    sc_nn = pipeline.createImageManip()
    if PERSON_DETECTION_NN == "pedestrian-detection-adas-0002":
        sc_nn.initialConfig.setResize(672, 384)
    if PERSON_DETECTION_NN == "person-detection-retail-0013":
        sc_nn.initialConfig.setResize(544, 320)
    sc_block.outputs['host'].link(sc_nn.inputImage)

    pers_nn = pipeline.createMobileNetDetectionNetwork()
    if PERSON_DETECTION_NN == "pedestrian-detection-adas-0002":
        if USE_BLOBCONVERTER:
            pers_nn.setBlobPath(blobconverter.from_zoo(
                name="pedestrian-detection-adas-0002",
                shaves=6,
                ))
        else:
            pers_nn.setBlobPath(str((Path(__file__).parent / Path('pedestrian-detection-adas-0002_openvino_2021.4_6shave.blob')).resolve().absolute()))
    if PERSON_DETECTION_NN == "person-detection-retail-0013":
        if USE_BLOBCONVERTER:
            pers_nn.setBlobPath(blobconverter.from_zoo(
                name="person-detection-retail-0013",
                shaves=6,
                ))
        else:
            pers_nn.setBlobPath(str((Path(__file__).parent / Path('person-detection-retail-0013_openvino_2021.4_6shave.blob')).resolve().absolute()))
    pers_nn.setConfidenceThreshold(0.5)
    sc_nn.out.link(pers_nn.input)

    xout_nn = pipeline.createXLinkOut()
    xout_nn.setStreamName("pers")
    pers_nn.out.link(xout_nn.input)

# Pipeline is now finished, and we need to find an available device to run our pipeline
# we are using context manager here that will dispose the device after we stop using it
with depthai.Device(pipeline) as device:
    q_host = device.getOutputQueue("host", maxSize=4, blocking=False)

    if TRY_PERSON_DETECTION:
        q_nn = device.getOutputQueue("pers", maxSize=4, blocking=False)

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    while True:
        in_host = q_host.tryGet()

        if TRY_PERSON_DETECTION:
            in_pers = q_nn.tryGet()

        if in_host is not None:
            frame = in_host.getCvFrame()

            if TRY_PERSON_DETECTION:
                if in_pers is not None:
                    for detection in in_pers.detections:
                        bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            
            cv2.imshow("host", frame)

        if cv2.waitKey(1) == ord('q'):
            break
