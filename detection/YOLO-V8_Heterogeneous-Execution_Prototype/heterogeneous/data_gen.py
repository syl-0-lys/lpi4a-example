import numpy as np

t = np.ones([1, 66, 8400], dtype=np.float32)
np.savez("test.npz", _model_22_Concat_3_output_0 = t)
