from imageai.Detection.Custom import CustomObjectDetection
import os
class Phone:
    def __init__(self, phone_model_path,detection_config_path,res_folder):
        self.phone_model_path = phone_model_path
        self.detection_config_path = detection_config_path
        self.res_folder = res_folder
    def detect_phone(self,img):
        flag=False
        file_name = img[img.rfind("\\")+1:]
        detector = CustomObjectDetection()
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath(self.phone_model_path)
        detector.setJsonPath(self.detection_config_path)
        detector.loadModel()
        dst = file_name[:file_name.rfind(".")]+ "_phone"+".png"
        dst = self.res_folder+dst
        detections = detector.detectObjectsFromImage(input_image=img, output_image_path=dst)
        if len(detections)>1:
            flag=True
            for detection in detections:
                print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
        elif len(detections) == 1:
            if round(detection["percentage_probability"]) > 68 :
                flag = True
            else:
                os.remove(dst)
        else:
            os.remove(dst)
        return flag
