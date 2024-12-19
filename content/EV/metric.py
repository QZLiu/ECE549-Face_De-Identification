import face_recognition
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
import dlib
import sys
import os
from collections import defaultdict

# 计算身份相似度（余弦相似度）
def cosine_similarity(encoding_a, encoding_b):
    return np.dot(encoding_a, encoding_b) / (np.linalg.norm(encoding_a) * np.linalg.norm(encoding_b))

# 关键点检测函数
def face_landmarks(image_path, detector, predictor):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        return [(p.x, p.y) for p in landmarks.parts()]
    return None

# 计算关键点误差
def landmark_error(landmarks1, landmarks2):
    return sum(np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in zip(landmarks1, landmarks2)) / len(landmarks1)

def psnr(orig, swap):
    original_image = face_recognition.load_image_file(orig)
    scores = {}
    for i, swp in enumerate(swap):
        swapped_image = face_recognition.load_image_file(swp)
        original_encoding = face_recognition.face_encodings(original_image)[0]
        score = 0
        try:
            encoding = face_recognition.face_encodings(swapped_image)[0]
            score = cosine_similarity(original_encoding, encoding)
        except:
            pass
        scores[i] = score
        print(f"Image {i} Identity Similarity: ", score)
    return scores

def ssim_(orig, swap):
    original_image = face_recognition.load_image_file(orig)
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) 
    scores = {}
    for i, swp in enumerate(swap):
        swapped_image = face_recognition.load_image_file(swp)
        swap_gray = cv2.cvtColor(swapped_image, cv2.COLOR_BGR2GRAY)
        score = ssim(original_gray, swap_gray)
        scores[i] = score
        print(f"SSIM for Swap {i}: ", score)

    return scores
        
def lpips(orig, swap):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
    
    landmarks_orig = face_landmarks(orig, detector, predictor)
    scores = {}
    for i, swp in enumerate(swap):
        landmarks = face_landmarks(swp, detector, predictor)
        score = landmark_error(landmarks_orig, landmarks)
        scores[i] = score
        print(f"Landmark Error Swap {i}: ", score)
    return scores

if __name__ == "__main__":
    orig = sys.argv[1]
    swap = []
    swpdir = '../SwpTmp/'
    flen = len(os.listdir(swpdir))
    flen = 24
    for i in range(flen-6, flen):
        file = f'{swpdir}result_{i}.jpg'
        swap.append(file)
    
    s1 = psnr(orig, swap)
    s2 = ssim_(orig, swap)
    s3 = lpips(orig, swap)
    
    dicts = [s1, s2, s3]
    def rank_dict(d):
        # Sort keys by value (higher value gets better rank, i.e., 1, 2, 3)
        return {k: rank for rank, (k, _) in enumerate(sorted(d.items(), key=lambda x: x[1], reverse=True), 1)}

    ranked_dicts = [rank_dict(d) for d in dicts]

    # Step 2: Tally ranks for each key across all dictionaries
    tally = defaultdict(int)
    for ranked in ranked_dicts:
        for key, rank in ranked.items():
            tally[key] += rank

    # Step 3: Max pooling - Find the key with the highest cumulative rank
    max_key = max(tally, key=tally.get)  # Key with the highest rank total
    max_value = tally[max_key]

    # Step 4: Final ranking of all keys based on cumulative ranks
    final_ranking = sorted(tally.items(), key=lambda x: x[1], reverse=True)

    # Output results
    # print("Ranks for each dictionary:", ranked_dicts)
    # print("Cumulative ranks (votes):", tally)
    # print("Max pooled key:", max_key, "with cumulative rank:", max_value)
    # print("Final ranking:", final_ranking)
    
    # output
    selected_res = f'result_{flen-6+max_key}.jpg'
    cv2.imwrite('../output/'+selected_res, cv2.imread(swpdir+selected_res))
    
        