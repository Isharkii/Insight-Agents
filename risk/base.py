"""
risk/base.py

Abstract base interface for risk scoring models.
All risk model implementations must inherit from BaseRiskModel.
"""

from abc import ABC, abstractmethod


class BaseRiskModel(ABC):
    """Abstract base class for risk scoring models.

    Defines the interface that all risk model implementations
    must follow. Enforces a consistent compute contract across
    different risk scoring strategies.
    """

    @abstractmethod
    def compute(self, inputs: dict) -> float:
        """Compute a risk score from the given inputs.

        Args:
            inputs: A dictionary containing the input variables
                    required by the specific risk model implementation.

        Returns:
            A float representing the computed risk score.
            Interpretation of the score range is defined by
            the implementing subclass.

        Raises:
            NotImplementedError: If the subclass does not implement
                                 this method.
            ValueError: If required keys are missing from inputs
                        (implementation-specific).
        """
        raise NotImplementedError("Subclasses must implement compute()")
