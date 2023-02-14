import contextlib
import joblib
from tqdm import tqdm

# taken from: https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/58936697#58936697

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
        

def test_tqdm_joblib():
    from math import sqrt
    from joblib import Parallel, delayed

    with tqdm_joblib(tqdm(desc="My calculation", total=10)) as progress_bar:
        Parallel(n_jobs=16)(delayed(sqrt)(i**2) for i in range(10))

if __name__ == "__main__":
    test_tqdm_joblib()