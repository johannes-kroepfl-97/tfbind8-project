import os
import numpy as np

# from design_bench.datasets.discrete_dataset import DiscreteDataset
# from design_bench.disk_resource import DiskResource, SERVER_URL
'''
import time
for _ in range(5):
    try:
        from design_bench.datasets.discrete_dataset import DiscreteDataset
        break
    except ValueError as e:
        time.sleep(1)
'''

from design_bench.datasets.discrete_dataset import DiscreteDataset
from design_bench.disk_resource import DiskResource, SERVER_URL

TF_BIND_8_FILES = ['tf_bind_8-SIX6_REF_R1\\tf_bind_8-x-0.npy', 'tf_bind_8-SIX6_REF_R1\\tf_bind_8-y-0.npy']

class TFBind8Dataset(DiscreteDataset):
    """A polypeptide synthesis dataset that defines a common set of functions
    and attributes for a model-based optimization dataset, where the
    goal is to find a design 'x' that maximizes a prediction 'y':

    max_x { y = f(x) }

    Public Attributes:

    name: str
        An attribute that specifies the name of a model-based optimization
        dataset, which might be used when labelling plots in a diagram of
        performance in a research paper using design-bench
    x_name: str
        An attribute that specifies the name of designs in a model-based
        optimization dataset, which might be used when labelling plots
        in a visualization of performance in a research paper
    y_name: str
        An attribute that specifies the name of predictions in a model-based
        optimization dataset, which might be used when labelling plots
        in a visualization of performance in a research paper

    x: np.ndarray
        the design values 'x' for a model-based optimization problem
        represented as a numpy array of arbitrary type
    input_shape: Tuple[int]
        the shape of a single design values 'x', represented as a list of
        integers similar to calling np.ndarray.shape
    input_size: int
        the total number of components in the design values 'x', represented
        as a single integer, the product of its shape entries
    input_dtype: np.dtype
        the data type of the design values 'x', which is typically either
        floating point or integer (np.float32 or np.int32)

    y: np.ndarray
        the prediction values 'y' for a model-based optimization problem
        represented by a scalar floating point value per 'x'
    output_shape: Tuple[int]
        the shape of a single prediction value 'y', represented as a list of
        integers similar to calling np.ndarray.shape
    output_size: int
        the total number of components in the prediction values 'y',
        represented as a single integer, the product of its shape entries
    output_dtype: np.dtype
        the data type of the prediction values 'y', which is typically a
        type of floating point (np.float32 or np.float16)

    dataset_size: int
        the total number of paired design values 'x' and prediction values
        'y' in the dataset, represented as a single integer
    dataset_distribution: Callable[np.ndarray, np.ndarray]
        the target distribution of the model-based optimization dataset
        marginal p(y) used for controlling the sampling distribution
    dataset_max_percentile: float
        the percentile between 0 and 100 of prediction values 'y' above
        which are hidden from access by members outside the class
    dataset_min_percentile: float
        the percentile between 0 and 100 of prediction values 'y' below
        which are hidden from access by members outside the class
    dataset_max_output: float
        the specific cutoff threshold for prediction values 'y' above
        which are hidden from access by members outside the class
    dataset_min_output: float
        the specific cutoff threshold for prediction values 'y' below
        which are hidden from access by members outside the class

    internal_batch_size: int
        the integer number of samples per batch that is used internally
        when processing the dataset and generating samples
    freeze_statistics: bool
        a boolean indicator that when set to true prevents methods from
        changing the normalization and sub sampling statistics

    is_normalized_x: bool
        a boolean indicator that specifies whether the design values
        in the dataset are being normalized
    x_mean: np.ndarray
        a numpy array that is automatically calculated to be the mean
        of visible design values in the dataset
    x_standard_dev: np.ndarray
        a numpy array that is automatically calculated to be the standard
        deviation of visible design values in the dataset

    is_normalized_y: bool
        a boolean indicator that specifies whether the prediction values
        in the dataset are being normalized
    y_mean: np.ndarray
        a numpy array that is automatically calculated to be the mean
        of visible prediction values in the dataset
    y_standard_dev: np.ndarray
        a numpy array that is automatically calculated to be the standard
        deviation of visible prediction values in the dataset

    is_logits: bool (only supported for a DiscreteDataset)
        a value that indicates whether the design values contained in the
        model-based optimization dataset have already been converted to
        logits and need not be converted again

    Public Methods:

    iterate_batches(batch_size: int, return_x: bool,
                    return_y: bool, drop_remainder: bool)
                    -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model
    iterate_samples(return_x: bool, return_y: bool):
                    -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        Returns an object that supports iterations, which yields tuples of
        design values 'x' and prediction values 'y' from a model-based
        optimization data set for training a model

    subsample(max_samples: int,
              max_percentile: float,
              min_percentile: float):
        a function that exposes a subsampled version of a much larger
        model-based optimization dataset containing design values 'x'
        whose prediction values 'y' are skewed
    relabel(relabel_function:
            Callable[[np.ndarray, np.ndarray], np.ndarray]):
        a function that accepts a function that maps from a dataset of
        design values 'x' and prediction values y to a new set of
        prediction values 'y' and relabels the model-based optimization dataset

    clone(subset: set, shard_size: int,
          to_disk: bool, disk_target: str, is_absolute: bool):
        Generate a cloned copy of a model-based optimization dataset
        using the provided name and shard generation settings; useful
        when relabelling a dataset buffer from the dis
    split(fraction: float, subset: set, shard_size: int,
          to_disk: bool, disk_target: str, is_absolute: bool):
        split a model-based optimization data set into a training set and
        a validation set allocating 'fraction' of the data set to the
        validation set and the rest to the training set

    normalize_x(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point design values 'x'
        as input and standardizes them so that they have zero
        empirical mean and unit empirical variance
    denormalize_x(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point design values 'x'
        as input and undoes standardization so that they have their
        original empirical mean and variance
    normalize_y(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point prediction values 'y'
        as input and standardizes them so that they have zero
        empirical mean and unit empirical variance
    denormalize_y(new_x: np.ndarray) -> np.ndarray:
        a helper function that accepts floating point prediction values 'y'
        as input and undoes standardization so that they have their
        original empirical mean and variance

    map_normalize_x():
        a destructive function that standardizes the design values 'x'
        in the class dataset in-place so that they have zero empirical
        mean and unit variance
    map_denormalize_x():
        a destructive function that undoes standardization of the
        design values 'x' in the class dataset in-place which are expected
        to have zero  empirical mean and unit variance
    map_normalize_y():
        a destructive function that standardizes the prediction values 'y'
        in the class dataset in-place so that they have zero empirical
        mean and unit variance
    map_denormalize_y():
        a destructive function that undoes standardization of the
        prediction values 'y' in the class dataset in-place which are
        expected to have zero empirical mean and unit variance

    --- for discrete tasks only

    to_logits(np.ndarray) > np.ndarray:
        A helper function that accepts design values represented as a numpy
        array of integers as input and converts them to floating point
        logits of a certain probability distribution
    to_integers(np.ndarray) > np.ndarray:
        A helper function that accepts design values represented as a numpy
        array of floating point logits as input and converts them to integer
        representing the max of the distribution

    map_to_logits():
        a function that processes the dataset corresponding to this
        model-based optimization problem, and converts integers to a
        floating point representation as logits
    map_to_integers():
        a function that processes the dataset corresponding to this
        model-based optimization problem, and converts a floating point
        representation as logits to integers

    """

    name = "tf_bind_8/tf_bind_8"
    y_name = "enrichment_score"
    x_name = "dna_sequence"

    @staticmethod
    def register_x_shards(transcription_factor='SIX6_REF_R1'):
        """Registers a remote file for download that contains design values
        in a format compatible with the dataset builder class;
        these files are downloaded all at once in the dataset initialization

        Arguments:

        transcription_factor: str
            a string argument that specifies which transcription factor to
            select for model-based optimization, where the goal is to find
            a length 8 polypeptide with maximum binding affinity

        Returns:

        resources: list of RemoteResource
            a list of RemoteResource objects specific to this dataset, which
            will be automatically downloaded while the dataset is built
            and may serve as shards if the dataset is large

        """
        
        output_list = []
        
        for file in TF_BIND_8_FILES:
            try:
                if transcription_factor in file:
                    print(f'{transcription_factor} is in file')
                    output_list.append(DiskResource(
                        file, is_absolute=False,
                        download_target=f"{SERVER_URL}/{file}",
                        download_method="direct"))
            except:
                continue

        '''return [DiskResource(
            file, is_absolute=False,
            download_target=f"{SERVER_URL}/{file}",
            download_method="direct") for file in TF_BIND_8_FILES
            if transcription_factor in file]'''
                                       
        return output_list

    @staticmethod
    def register_y_shards(transcription_factor='SIX6_REF_R1'):
        """Registers a remote file for download that contains prediction
        values in a format compatible with the dataset builder class;
        these files are downloaded all at once in the dataset initialization

        Arguments:

        transcription_factor: str
            a string argument that specifies which transcription factor to
            select for model-based optimization, where the goal is to find
            a length 8 polypeptide with maximum binding affinity

        Returns:

        resources: list of RemoteResource
            a list of RemoteResource objects specific to this dataset, which
            will be automatically downloaded while the dataset is built
            and may serve as shards if the dataset is large

        """

        '''return [DiskResource(
            file.replace("-x-", "-y-"), is_absolute=False,
            download_target=f"{SERVER_URL}/{file.replace('-x-', '-y-')}",
            download_method="direct") for file in TF_BIND_8_FILES
            if transcription_factor in file]'''

        output_list = []
        
        for file in TF_BIND_8_FILES:
            try:
                if transcription_factor in file:
                    print(f'{transcription_factor} is in file')
                    output_list.append(DiskResource(
                        file.replace("-x-", "-y-"), is_absolute=False,
                        download_target=f"{SERVER_URL}/{file.replace('-x-', '-y-')}",
                        download_method="direct"))
            except:
                continue
        return output_list
    
    '''                                   
    def __init__(self, soft_interpolation=0.6,
                 transcription_factor='SIX6_REF_R1', **kwargs):
        """Initialize a model-based optimization dataset and prepare
        that dataset by loading that dataset from disk and modifying
        its distribution

        Arguments:

        soft_interpolation: float
            a floating point hyper parameter used when converting design values
            from integers to a floating point representation as logits, which
            interpolates between a uniform and dirac distribution
            1.0 = dirac, 0.0 -> uniform
        transcription_factor: str
            a string argument that specifies which transcription factor to
            select for model-based optimization, where the goal is to find
            a length 8 polypeptide with maximum binding affinity
        **kwargs: dict
            additional keyword arguments which are used to parameterize the
            data set generation process, including which shard files are used
            if multiple sets of data set shard files can be loaded

        """

        # set the names the describe the dataset
        self.name = f"tf_bind_8-{transcription_factor}/tf_bind_8"
        self.y_name = "enrichment_score"
        self.x_name = "dna_sequence"

        # initialize the dataset using the method in the base class
        super(TFBind8Dataset, self).__init__(
            self.register_x_shards(transcription_factor=transcription_factor),
            self.register_y_shards(transcription_factor=transcription_factor),
            is_logits=False, num_classes=4,
            soft_interpolation=soft_interpolation, **kwargs)
    '''
    
    def __init__(self, base_path, transcription_factor='SIX6_REF_R1', soft_interpolation=0.6, **kwargs):
        """
        Load the dataset from a user-specified local path.

        Args:
            base_path (str): Root directory containing subfolders per transcription factor,
                             each with 'tf_bind_8-x-0.npy' and 'tf_bind_8-y-0.npy'.
            transcription_factor (str): Name of the TF to load.
            soft_interpolation (float): Logit interpolation factor.
        """
        self.name = f"tf_bind_8-{transcription_factor}/tf_bind_8"
        self.y_name = "enrichment_score"
        self.x_name = "dna_sequence"

        folder_path = os.path.join(base_path, f"tf_bind_8-{transcription_factor}")
        x_path = os.path.join(folder_path, "tf_bind_8-x-0.npy")
        y_path = os.path.join(folder_path, "tf_bind_8-y-0.npy")

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            raise FileNotFoundError(f"Missing x or y .npy files for transcription factor '{transcription_factor}' in {folder_path}")

        x_data = np.load(x_path)
        y_data = np.load(y_path)

        super().__init__(
            [x_data], [y_data],
            is_logits=False,
            num_classes=4,
            soft_interpolation=soft_interpolation,
            **kwargs
        )