import dlib
import numpy as np
import cv2
import math

class face_detector():
    def __init__(self, image):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
        self.POINTS_NUM_LANDMARK = 68
        self.image = cv2.imread(path_image)

    def rect_to_bb(self,rect): 
        x = rect.left()
        y = rect.top()
        w = rect.right() - x
        h = rect.bottom() - y
        return (x, y, w, h)

    def get_bbox(self, show = False):
        img = self.image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = self.detector(gray, 1)
        print("Number of faces detected: {}".format(len(dets)))

        for (i, rect) in enumerate(dets):
            shape = self.predictor(gray, rect)
            shape = np.matrix([[p.x, p.y] for p in shape.parts()])
            (x, y, w, h) = self.rect_to_bb(rect)
            
        if show:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("bbox", img)
            cv2.waitKey(0)
        
        return (x, y, w, h)

    def get_landmark(self, show = False):
        img = self.image.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = self.detector(gray, 1)

        #just 1 face
        det = dets[0]
        shape = self.predictor(img, det)
        
        landmark = np.matrix([[p.x, p.y] for p in shape.parts()])

        if show :
            self.show_landmark(landmark)
        return landmark
        
    def show_landmark(self, landmark):
        img = self.image.copy()
        points_list = landmark.A
        for point in points_list:
            cv2.circle(img, tuple(point), 1, (0, 255, 0), -1)
        cv2.imshow("landmark", img)
        cv2.waitKey(0)

    def pose_estimation(self, show = False):
        img = self.image.copy()
        size = img.shape
        landmark = self.get_landmark()
        points_list = landmark.A

        #2D image points.
        image_points = np.array([
                                    points_list[30],     # Nose tip **
                                    points_list[8],      # Chin
                                    points_list[36],     # Left eye left corner **
                                    points_list[45],     # Right eye right corne **
                                    points_list[48],     # Left Mouth corner **
                                    points_list[54]      # Right mouth corner **
                                ], dtype="double")

        # 3D model points.
        model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])
        
        # Camera internals
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                                [[focal_length, 0, center[0]],
                                [0, focal_length, center[1]],
                                [0, 0, 1]], dtype = "double"
                                )

        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        # print("Rotation Vector:\n {0}".format(rotation_vector))
        # print("Translation Vector:\n {0}".format(translation_vector))

        if show:
            # Project a 3D point onto the image plane.
            # We use this to draw a line sticking out of the nose

            axis = np.float32([[500,0,0], [0,500,0], [0,0,500]])
            (nose_end_point, jacobian) = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    
            
            for p in image_points:
                cv2.circle(img, (int(p[0]), int(p[1])), 2, (0,0,255), -1)

            p0 = (int(image_points[0][0]), int(image_points[0][1]))

            p1 = (int(nose_end_point[0][0][0]), int(nose_end_point[0][0][1]))
            p2 = (int(nose_end_point[1][0][0]), int(nose_end_point[1][0][1]))
            p3 = (int(nose_end_point[2][0][0]), int(nose_end_point[2][0][1]))
            
            cv2.line(img, p0, p1, (0,255,0), 1)
            cv2.line(img, p0, p2, (0,255,0), 1)
            cv2.line(img, p0, p3, (0,255,0), 1)
            
            # Display image
            cv2.imshow("pose", img)
            cv2.waitKey(0)

        return rotation_vector, translation_vector
        
    def toEuler(self, rotation_vector, translation_vector):
        rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

        proj_matrix = np.hstack((rvec_matrix, translation_vector))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 
        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))

        return pitch, roll, yaw


if __name__ == '__main__':
    path_image = "image/test.png"
    model = face_detector(path_image) 

    bbox = model.get_bbox(show = True)
    landmark = model.get_landmark(show = True)
    pose = model.pose_estimation(show = True)()