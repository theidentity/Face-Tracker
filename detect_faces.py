import face_recognition
from glob import glob
from PIL import Image


def crop_faces(img):
	face_locations = face_recognition.face_locations(img,model="cnn")
	faces = []
	for location in face_locations:
		top,right,bottom,left = location
		face = img[top:bottom,left:right,:]
		faces.append(face)
	return faces

def is_same_face(img,target):
	img_encoding = face_recognition.face_encodings(img)[0]
	target_encoding = face_recognition.face_encodings(target)[0]
	results = face_recognition.compare_faces([img_encoding], target_encoding)
	return results[0]


# modi = face_recognition.load_image_file('bad/images431.jpeg')
# manmohan = face_recognition.load_image_file('good/images252.jpeg')

# modi = crop_faces(modi)[0]
# manmohan = crop_faces(manmohan)[0]

# pil_image = Image.fromarray(modi)
# pil_image.show()
# pil_image = Image.fromarray(manmohan)
# pil_image.show()

# print(is_same_face(modi,manmohan))
