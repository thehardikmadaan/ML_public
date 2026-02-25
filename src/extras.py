#add extras to this file
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def monkey_patch_get_signature_names_out():
    """Monkey patch some classes which did not handle get_feature_names_out()
       correctly in Scikit-Learn 1.0.*."""
    from inspect import Signature, signature, Parameter
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    default_get_feature_names_out = StandardScaler.get_feature_names_out

    if not hasattr(SimpleImputer, "get_feature_names_out"):
      print("Monkey-patching SimpleImputer.get_feature_names_out()")
      SimpleImputer.get_feature_names_out = default_get_feature_names_out

    if not hasattr(FunctionTransformer, "get_feature_names_out"):
        print("Monkey-patching FunctionTransformer.get_feature_names_out()")
        orig_init = FunctionTransformer.__init__
        orig_sig = signature(orig_init)

        def __init__(*args, feature_names_out=None, **kwargs):
            orig_sig.bind(*args, **kwargs)
            orig_init(*args, **kwargs)
            args[0].feature_names_out = feature_names_out

        __init__.__signature__ = Signature(
            list(signature(orig_init).parameters.values()) + [
                Parameter("feature_names_out", Parameter.KEYWORD_ONLY)])

        def get_feature_names_out(self, names=None):
            if callable(self.feature_names_out):
                return self.feature_names_out(self, names)
            assert self.feature_names_out == "one-to-one"
            return default_get_feature_names_out(self, names)

        FunctionTransformer.__init__ = __init__
        FunctionTransformer.get_feature_names_out = get_feature_names_out

monkey_patch_get_signature_names_out()


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds, target_precision=0.90, save_name=None):
    """
    Plots precision and recall against thresholds, highlighting a specific target precision.
    """
    # Create the figure
    plt.figure(figsize=(8, 4))

    # Plot the main Precision and Recall curves
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)

    # Add labels, legend, grid, and axis limits
    plt.legend(loc="center right", fontsize=16)
    plt.xlabel("Threshold", fontsize=16)
    plt.grid(True)
    plt.axis([-50000, 50000, 0, 1])

    # Calculate the threshold and recall that match the target precision
    # np.argmax finds the first index where the precision meets or exceeds the target
    idx = np.argmax(precisions >= target_precision)
    target_threshold = thresholds[idx]
    target_recall = recalls[idx]

    # Draw red dotted lines to point out the specific threshold and recall
    plt.plot([target_threshold, target_threshold], [0., target_precision], "r:")
    plt.plot([-50000, target_threshold], [target_precision, target_precision], "r:")
    plt.plot([-50000, target_threshold], [target_recall, target_recall], "r:")

    # Draw red dots on the exact intersections
    plt.plot([target_threshold], [target_precision], "ro")
    plt.plot([target_threshold], [target_recall], "ro")

    # Save the figure if a name is provided
    if save_name:
        plt.savefig(f"{save_name}.png", bbox_inches="tight")

    # Display the plot
    plt.show()

# Example of how to call this function :
# plot_precision_recall_vs_threshold(precisions, recalls, thresholds, target_precision=0.90, save_name="precision_recall_vs_threshold_plot")