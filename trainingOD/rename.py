# %%
trainpath = "../../../pathcore-inspection/mvtec_anomaly_detection/bean/train/good"
testpath_good = "../../../pathcore-inspection/mvtec_anomaly_detection/bean/test/good"
testpath_tear = "../../../pathcore-inspection/mvtec_anomaly_detection/bean/test/tear"
ground_truth = "../../../pathcore-inspection/mvtec_anomaly_detection/bean/ground_truth/tear"

import os
try:
      print(f"""
            
            Train    : {(os.listdir(trainpath))}
            Test good: {(os.listdir(testpath_good))}
            Test tear: {(os.listdir(testpath_tear))}
            """)

      print(f"""
            Train    : {len(os.listdir(trainpath))}
            Test good: {len(os.listdir(testpath_good))}
            Test tear: {len(os.listdir(testpath_tear))}
            """)
except Exception as e:
      print(e)
      print("Error")
      pass

# %%
def find_number(name, path = ground_truth):
    if path == ground_truth:
        # print(name.split("_"))
        return name.split("_")[0]
    else:
        return int(name.split(".")[0])
    
print(find_number("164_0.png", path=trainpath))

def sorted_files(path):
    if path == ground_truth: 
        new = [find_number(name) for name in os.listdir(path)]
    else:
        new = [find_number(name, path) for name in os.listdir(path)]
    new.sort()
    if path == ground_truth:
        new = [f"{i:04}_mask.png" for i in new]
    else:
        new = [f"{i:04}.png" for i in new]
    return new

sorted_trainpath = sorted_files(trainpath)
sorted_testpath_good = sorted_files(testpath_good)
sorted_testpath_tear = sorted_files(testpath_tear)

print(f"""
    Train    : {sorted_trainpath[:5]}
    Test good: {sorted_testpath_good[:5]}
    Test tear: {sorted_testpath_tear[:5]}
      """)

print(f"""
    Train    : {len(sorted_trainpath)}
    Test good: {len(sorted_testpath_good)}
    Test tear: {len(sorted_testpath_tear)}
      """)

# %%
def renamer(path):
    if path == trainpath:
        for i in range(len(sorted_trainpath)):
            # print(os.path.join(path, sorted_trainpath[i]), os.path.join(path, f"{i:03}.png"))
            os.rename(os.path.join(path, sorted_trainpath[i]), os.path.join(path, f"{i:04}.png"))
    elif path == testpath_good:
        for i in range(len(sorted_testpath_good)):
            # print(os.path.join(path, sorted_testpath_good[i]), os.path.join(path, f"{i:03}.png"))
            os.rename(os.path.join(path, sorted_testpath_good[i]), os.path.join(path, f"{i:04}.png"))
    elif path == testpath_tear:
        for i in range(len(sorted_testpath_tear)):
            # print(os.path.join(path, sorted_testpath_tear[i]), os.path.join(path, f"{i:03}.png"))
            os.rename(os.path.join(path, sorted_testpath_tear[i]), os.path.join(path, f"{i:04}.png"))
            
try:
    renamer(trainpath)
    renamer(testpath_good)
    renamer(testpath_tear)
    renamer(ground_truth)
except Exception as e:
    print(e)
    print("Files already renamed")

print(f"""
    Train    : {len(os.listdir(trainpath))}
    Test good: {len(os.listdir(testpath_good))}
    Test tear: {len(os.listdir(testpath_tear))}
      """)

# %%
import os

print(f"""
      
      Train    : {(os.listdir(trainpath))}
      Test good: {(os.listdir(testpath_good))}
      Test tear: {(os.listdir(testpath_tear))}
      """)

print(f"""
      Train    : {len(os.listdir(trainpath))}
      Test good: {len(os.listdir(testpath_good))}
      Test tear: {len(os.listdir(testpath_tear))}
      """)

# %%



