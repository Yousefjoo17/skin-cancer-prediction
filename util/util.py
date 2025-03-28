
import datetime
import time

from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

#importstr('math', 'sqrt')  # Returns math.sqrt
#importstr('training', 'SkinCancerClsTrainingApp')  # Returns training.SkinCancerClsTrainingApp

def importstr(module_str, from_=None): # eg: module_str = 'p2ch11.training' and from_ = 'LunaTrainingApp'
    """
    >>> importstr('os')
    <module 'os' from '.../os.pyc'>
    >>> importstr('math', 'fabs')
    <built-in function fabs>
    """
    if from_ is None and ':' in module_str:
        module_str, from_ = module_str.rsplit(':')

    module = __import__(module_str)
    for sub_str in module_str.split('.')[1:]:
        module = getattr(module, sub_str)

    if from_:
        try:
            return getattr(module, from_)
        except:
            raise ImportError('{}.{}'.format(module_str, from_))
    return module

def enumerateWithEstimate(
        iterable,
        description, # Epoch number "training", "validation" or "testing" 
        start_ndx=0,
        total_batches=None,
):
    if total_batches is None:
        total_batches = len(iterable) # number of batches

    log.warning("{} ----/{}, starting".format(
        description,
        total_batches,
    ))

    start_time = time.time()

    for (batch_index, batch_data) in enumerate(iterable):
        yield (batch_index, batch_data)

        if batch_index % 25 == 0 :
            elapsed_time = (time.time() - start_time)
            estimated_total_time = (elapsed_time / (batch_index - start_ndx + 1)) * (total_batches - start_ndx)
            estimated_completion_time = datetime.datetime.fromtimestamp(start_time + estimated_total_time)
            remaining_time = datetime.timedelta(seconds=estimated_total_time)

            log.info("{} {:-4}/{}, expected completion at {}, remaining time {}".format(
                description,
                batch_index,
                total_batches,
                str(estimated_completion_time).rsplit('.', 1)[0],
                str(remaining_time).rsplit('.', 1)[0],
            ))


        if batch_index + 1 == start_ndx:
            start_time  = time.time()

    log.warning("{} ----/{}, done at {}".format(
        description,
        total_batches,
        str(datetime.datetime.now()).rsplit('.', 1)[0],
    ))



import matplotlib.pyplot as plt

def plot_tensor_image(img_tensor):
    img_tensor = img_tensor.squeeze(0)  # Remove batch dimension -> (3, 224, 224)
    img_numpy = img_tensor.permute(1, 2, 0).numpy()  # Convert to (224, 224, 3)

    # Normalize values to [0,1] if needed
    img_numpy = (img_numpy - img_numpy.min()) / (img_numpy.max() - img_numpy.min())

    plt.imshow(img_numpy)
    plt.axis('off')  # Hide axes
    plt.show()


