import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing


def apply_reweighing(X, y, protected_attr, privileged_val, unprivileged_val):
    df = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
    df['target'] = y
    df['protected'] = protected_attr
    dataset = BinaryLabelDataset(
        df=df,
        label_names=['target'],
        protected_attribute_names=['protected'],
        favorable_label=1,
        unfavorable_label=0
    )
    rw = Reweighing(
        unprivileged_groups=[{'protected': unprivileged_val}],
        privileged_groups=[{'protected': privileged_val}]
    )
    dataset_transf = rw.fit_transform(dataset)
    return dataset_transf.features, dataset_transf.labels.ravel(), dataset_transf.instance_weights
