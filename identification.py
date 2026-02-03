# identification.py
import cv2
import numpy as np

# ACCURACY FIX: Relaxed ratio slightly (0.7 -> 0.75) for rotated matching
def match_sift_flann(query_des, query_kp, db_des, db_kp, ratio_thresh=0.75):
    """
    Matches SIFT descriptors using FLANN and validates with Geometric RANSAC.
    """
    if query_des is None or db_des is None:
        return 0
    
    if len(query_des) < 2 or len(db_des) < 2:
        return 0

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # ACCURACY FIX: Higher checks = more precise nearest neighbor search
    search_params = dict(checks=100) 

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(query_des, db_des, k=2)
    except Exception:
        return 0

    # 1. Lowe's Ratio Test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
            
    # 2. Geometric Verification (RANSAC)
    MIN_MATCH_COUNT = 4 
    
    if len(good_matches) >= MIN_MATCH_COUNT:
        src_pts = np.float32([query_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([db_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # RANSAC threshold 5.0 is standard; increasing to 10.0 allows looser geometric fit
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if M is not None:
            matchesMask = mask.ravel().tolist()
            return sum(matchesMask) 
        else:
            return 0
    else:
        return 0

def find_top_matches_sift(query_sift_data, candidate_db, top_n=3):
    query_kp, query_des = query_sift_data
    results = []

    for path in candidate_db:
        entry = candidate_db[path]
        db_des = entry.get("sift_des")
        db_kp = entry.get("sift_kp") 
        
        if db_kp is None or db_des is None:
             continue

        score = match_sift_flann(query_des, query_kp, db_des, db_kp)
        
        # Only add if there is at least 1 verified match
        if score > 0:
            results.append((score, path))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_n]