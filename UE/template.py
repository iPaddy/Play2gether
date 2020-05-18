import cv2
import numpy as np

def load_obj(path):
    """
    Ad-hoc implementation to load OBJ files. Probably only works for bunny.obj
    @param path: path to a .obj file
    @return: Nx(x,y,z) array of positions, Mx(i0, i1, i2) array of triangle faces
    """
    vtx = []
    idx = []
    
    f = open(path)
    for l in f.readlines():
        if l[0] == "v": 
            vtx.append(l[1:].split())
        elif l[0] == "f":
            idx.append(l[1:].split())
    
    idx = np.int32(np.array(idx)) - 1
    vtx = np.float32(np.array(vtx)).reshape(-1, 1, 3)
    
    return vtx, idx

### YOUR CODE BEGINS HERE

# Task 1
vtx, idx = load_obj(".\\bunny.obj")
#vtx, idx = load_obj("C:\\Users\\Victoria\\Desktop\\Uni\\importantUNI\\SoSe20\\VRAR\\UE\\venv\\bunny.obj")

img = np.ones((480,640,3),np.uint8)*0
color = (0,0,255)
#cv2.circle(img,center=(640,480),radius=20,color=color,thickness=5)

# camera matrix K, rotation vector R and translation vector t
K = np.array([[800,0,320],[0,800,240],[0,0,1]],np.float)
R = np.array([np.pi/2, 0,0])
t = np.array([-40, 30, 100],np.float)

pts,_ = cv2.projectPoints(objectPoints=vtx,rvec=R,tvec=t,cameraMatrix=K,distCoeffs=np.empty(8))
pts = np.array([np.floor(p) for p in pts], np.int32)

for ind in idx:
    cv2.polylines(img=img,pts=[pts[ind]],isClosed=True,color=color,thickness=1,lineType=cv2.LINE_AA)

cv2.drawFrameAxes(image=img,cameraMatrix=K,distCoeffs=np.empty(4),rvec=R,tvec=t,length=10)
#cv2.imshow('VRAR Task 1',img)
#cv2.imwrite('task1.png',img)
#cv2.waitKey(0)

# Task 2
distCoeffs=np.array([-1,0,0,0],np.float)
img1 = np.ones((480,640,3),np.uint8)*0

pts,_ = cv2.projectPoints(objectPoints=vtx,rvec=R,tvec=t,cameraMatrix=K,distCoeffs=distCoeffs)
pts = np.array([np.floor(p) for p in pts], np.int32)

a = [ind for ind in idx if np.cross(pts[ind[1]]-pts[ind[0]],pts[ind[2]]-pts[ind[0]]) <= 0]
for ind in a:
    cv2.polylines(img=img1,pts=[pts[ind]],isClosed=True,color=color,thickness=1,lineType=cv2.LINE_AA)

cv2.drawFrameAxes(image=img1,cameraMatrix=K,distCoeffs=np.empty(4),rvec=R,tvec=t,length=10)
#cv2.imshow('VRAR Task 2',img1)
#cv2.imwrite('task2.png',img1)
cv2.waitKey(0)

### Task 3
img2 = np.ones((480,640,3),np.uint8)*0

visiableIdx = np.array([ind for ind in idx if np.cross(pts[ind[1]]-pts[ind[0]],pts[ind[2]]-pts[ind[0]]) <= 0])

# calculate normals
normals = np.zeros((np.shape(idx)[0],3))
i = 0
perVertex = False
perFace = not perVertex

if perFace:
     for ind in visiableIdx:
        normals[i] = np.cross(vtx[ind[2]]-vtx[ind[0]],vtx[ind[1]]-vtx[ind[0]])[0]
        i += 1
if perVertex:
    for ind in idx:
        normals[ind[0]] += np.cross(vtx[ind[2]]-vtx[ind[0]],vtx[ind[1]]-vtx[ind[0]])[0]
        normals[ind[1]] += np.cross(vtx[ind[0]]-vtx[ind[1]],vtx[ind[2]]-vtx[ind[1]])[0]
        normals[ind[2]] += np.cross(vtx[ind[1]]-vtx[ind[2]],vtx[ind[0]]-vtx[ind[2]])[0]

# transform normals in camera cs
R_mat,_ = cv2.Rodrigues(R)

normals = np.array([(R_mat @ xi)  for xi in normals if not (np.linalg.norm(xi) == 0.0)])
normals = np.array([xi/np.linalg.norm(xi) for xi in normals if not (np.linalg.norm(xi) == 0.0)])


pts,_ = cv2.projectPoints(objectPoints=vtx,rvec=R,tvec=t,cameraMatrix=K,distCoeffs=distCoeffs)
pts = np.array([np.floor(p) for p in pts], np.int32)

j = 0
for ind in visiableIdx:
    # d = z component of normal because of L=[0,0,1]
    if perVertex:
        d = ((normals[ind[0]]+normals[ind[1]]+normals[ind[2]])/np.linalg.norm(normals[ind[0]]+normals[ind[1]]+normals[ind[2]]))[2]
    if perFace:
        d = normals[j,2]/np.linalg.norm(normals[j,0]+normals[j,1]+normals[j,2])
        j += 1
    if d > 0:
        cv2.fillConvexPoly(img=img2,points=pts[ind],color=(d*255,0,d*255),lineType=cv2.LINE_AA)

cv2.imshow('VRAR Task 3',img2)
cv2.imwrite('task3.png',img2)
cv2.waitKey(0)