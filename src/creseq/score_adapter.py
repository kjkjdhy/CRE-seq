# src/creseq/score_adapter.py 追加 / 覆盖
from __future__ import annotations
import numpy as np

BASE_TO_IDX = {"A":0,"C":1,"G":2,"T":3}

def one_hot_encode(seqs: list[str]) -> np.ndarray:
    L = len(seqs[0])
    X = np.zeros((len(seqs), L, 4), dtype=np.float32)
    for i,s in enumerate(seqs):
        assert len(s)==L, "All sequences must have the same length"
        for j,ch in enumerate(s):
            X[i,j,BASE_TO_IDX[ch]] = 1.0
    return X

class DeepSTARRScorer:
    """
    通用 SavedModel 加载器：
    - 优先用 serving_default 签名；没有就直接调用可调用对象
    - 只在你传 --deepstarr 时才会 import tensorflow
    """
    def __init__(self, savedmodel_dir: str):
        try:
            import tensorflow as tf
        except Exception as e:
            raise ImportError(
                "TensorFlow is required for DeepSTARRScorer. "
                "Install with: python -m pip install tensorflow"
            ) from e
        self.tf = tf
        self.model = tf.saved_model.load(savedmodel_dir)
        self.fn = None
        # 取签名（兼容不同导出方式）
        if hasattr(self.model, "signatures") and self.model.signatures:
            self.fn = self.model.signatures.get("serving_default", None)

    def score_batch(self, seqs: list[str]) -> np.ndarray:
        tf = self.tf
        X = one_hot_encode(seqs)
        X = tf.constant(X)
        if self.fn is not None:
            out = self.fn(X)
            # 取第一个输出张量
            y = next(iter(out.values())).numpy()
        else:
            # 直接可调用（有些 SavedModel 这么导出）
            y = self.model(X).numpy()
        return y.squeeze().astype(np.float32)

# 已有的 DummyScorer 保留
class DummyScorer:
    def score_batch(self, seqs: list[str]) -> np.ndarray:
        L = len(seqs[0])
        gcs = np.array([(s.count("G")+s.count("C"))/L for s in seqs], dtype=np.float32)
        return 1.0 - np.abs(gcs - 0.5)
