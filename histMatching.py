import SimpleITK as sitk
import numpy as np

#use sitk image as input


def casttype(image, dtype):
    castFilter = sitk.CastImageFilter()
    castFilter.SetOutputPixelType(dtype)
    return castFilter.Execute(image)


def findLandmark(image, landmarkList):
    locations = []
    image_array = sitk.GetArrayFromImage(image)
    image_brain_array = np.array(image_array, dtype=np.float32)

    image_brain_array = image_brain_array[image_brain_array!=0]

    image_brain_array = np.sort(image_brain_array)
    #minimum = image_brain_array[math.ceil(len(image_brain_array)*landmarkList[0])]
    maximum = image_brain_array[math.floor(len(image_brain_array)*landmarkList[1])]
    #image_brain_array[image_brain_array<minimum] = 0
    #print(minimum, maximum)
    image_brain_array[image_brain_array>maximum] = maximum

    rescale_brain_array = image_brain_array[image_brain_array!=0]

    rescale_brain_array = (rescale_brain_array * 255) / maximum

    rescale_brain_array = np.sort(rescale_brain_array)
    for landmark in landmarkList[2:]:
        location = rescale_brain_array[math.ceil(len(rescale_brain_array)*landmark)]
        locations.append(location)
    return locations, 0, maximum


def hist_matching(image, landmarkList, landmarksLocations):
    image = casttype(image, sitk.sitkFloat32)
    case_landmarks, case_min, case_max = findLandmark(image, landmarkList)
    print(case_landmarks)
    case_landmarks = [round(x) for x in case_landmarks]
    image_array = sitk.GetArrayFromImage(image)
    image_array = np.array(image_array, dtype=np.float32)
    image_array[image_array>=case_max] = case_max
    image_array_rescale = (image_array * 255) / case_max
    tmp = np.array(image_array_rescale, np.float32)
    tmp[tmp>case_landmarks[0]] = 0
    tmp = ((tmp - 0) * landmarksLocations[0]) / (case_landmarks[0] - 0)
    array_histMatch = tmp
    
    for i in range(len(landmarkList)-3):
        tmp = np.array(image_array_rescale, np.float32)
        tmp[tmp<=case_landmarks[i]] = 0
        tmp[tmp>case_landmarks[i+1]] = 0
        empty_tmp = np.array(tmp, np.float32)
        empty_tmp[empty_tmp!=0] = 1
        tmp = (tmp * (landmarksLocations[i+1]-landmarksLocations[i]) + case_landmarks[i+1]*landmarksLocations[i] - case_landmarks[i]*landmarksLocations[i+1])/(case_landmarks[i+1]-case_landmarks[i])
        tmp = tmp * empty_tmp
        array_histMatch = array_histMatch + tmp

    tmp = np.asarray(image_array_rescale, np.float32)
    tmp[tmp<=case_landmarks[-1]] = 0
    empty_tmp = np.array(tmp, np.float32)
    empty_tmp[empty_tmp!=0] = 1
    tmp = (tmp * (255-landmarksLocations[-1]) + 255*landmarksLocations[-1] - case_landmarks[-1]*255)/(255-case_landmarks[-1])
    tmp = tmp * empty_tmp
    array_histMatch = array_histMatch + tmp

    img_histMatch = sitk.GetImageFromArray(array_histMatch)
    img_histMatch.SetDirection(image.GetDirection())
    img_histMatch.SetOrigin(image.GetOrigin())
    img_histMatch.SetSpacing(image.GetSpacing())
    return img_histMatch



landmarkList = (0,0.9995,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95)
landmarksLocations = (29.0, 50.0, 72.0, 90.0, 102.0, 109.0, 114.0, 119.0, 122.0, 126.0, 129.0, 132.0, 136.0, 139.0, 143.0, 147.0, 152.0, 159.0, 169.0)