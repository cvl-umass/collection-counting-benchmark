"""Analysis layer: read run artifacts, simulate fixed-model trajectories, and apply the estimators.

``trajectory``     load a run's artifacts into the arrays the estimators consume.
``simulate``       fixed-model-trick simulator.
``run_estimators`` apply every estimator to a trajectory -> tidy DataFrame.
"""
