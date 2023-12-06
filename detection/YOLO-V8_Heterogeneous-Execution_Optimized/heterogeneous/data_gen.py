import numpy as np

t = np.ones([1, 66, 6400], dtype=np.float32)
np.savez("test1.npz", _model_22_Reshape_output_0 = t)

t1 = np.ones([1, 192, 40, 40], dtype=np.float32)
np.savez("test2.npz", _model_18_Concat_output_0 = t1)

t2 = np.ones([1, 256, 20, 20], dtype=np.float32)
np.savez("test3.npz", _model_9_cv2_act_Mul_output_0 = t2)
