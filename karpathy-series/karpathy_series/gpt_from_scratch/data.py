from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class SequenceData:
    training: Tensor
    validation: Tensor
    testing: Tensor


@dataclass(frozen=True)
class SequenceDataSplit:
    training: float
    validation: float
    testing: float | None = None

    def __call__(self, data: Tensor) -> SequenceData:
        total = self.training + self.validation + (0 if self.testing is None else self.testing)
        training_fraction = self.training / total

        validation_split_index = int(training_fraction * len(data))
        training = data[:validation_split_index]

        if self.testing is None:
            testing_split_index = len(data)
        else:
            validation_fraction = self.validation / total
            testing_split_index = int(validation_fraction * len(data)) + validation_split_index
        validation = data[validation_split_index:testing_split_index]

        testing = data[testing_split_index:]
        return SequenceData(training, validation, testing)
