## Parameters

### Common Training Settings
| Name                        | Type           | Default | Applies to                        | Description                                                                                         |
|-----------------------------|----------------|---------|-----------------------------------|-----------------------------------------------------------------------------------------------------|
optimization steps. Increase to train longer.                                       |
| **NumberOfEpochs**          | `int`          | —       | All                               | Number of epochs to run over the dataset.                                                           |
| **batch_size**              | `int`          | 32      | All                               | Number of samples per gradient update.                                                             |
| **learning_rate**           | `float`        | 1e-4    | All                               | Initial step size for the optimizer.                                                  |
| **weight_decay**            | `float`        | 0       | All                               | L2 regularization strength.                                                         |
| **use_early_stopping**      | `bool`         | True    | All                               | Whether to stop training if validation loss stops improving.                                        |
| **patience**                | `int`          | 10      | All                               | Number of epochs with no improvement before stopping (if `use_early_stopping=True`).                |
| **min_delta**               | `float`        | 1e-4    | All                               | Minimum change in monitored loss to qualify as improvement.                                         |
| **iterations**              | `int`          | 2       | All                               | Number of repeated runs (for averaging results).                                                    |
| **print_epoch_stats**       | `bool`         | False   | All                               | If True, prints metrics each epoch.                                                                   |

### Data Augmentation & Preprocessing
| Name                       | Type            | Default          | Applies to | Description                                                                                     |
|----------------------------|-----------------|------------------|------------|-------------------------------------------------------------------------------------------------|
| **augment_filter_fn**      | `callable/None` | `None`           | All        | Function `row→bool` deciding which rows to augment.  (lambda row: row.label > 0)                                            |
| **augment_filter_descrp**  | `str`           | `""`             | All        | Text description of `augment_filter_fn`.                                                         |
| **truncation**             | `bool`          | False            | All        | Whether to truncate the start of each signal.                                                   |
| **truncation_portion**     | `float`         | 0.3              | All        | Fraction of the beginning to remove (0≤portion<1).                                               |
| **oversample**             | `bool`          | False            | All        | Whether to duplicate under-represented rows.                                                     |
| **oversample_fn**          | `callable/None` | `None`           | All        | Function selecting rows for oversampling. (lambda row: row.label > 0)                                                       |
| **oversample_descrp**      | `str`           | `""`             | All        | Text description of `oversample_fn`.                                                             |
| **oversample_times**       | `float`         | 1                | All        | How many times to duplicate selected rows (can be fractional).                                   |
| **sliding**                | `bool`          | False            | All        | Whether to slice sequences into sliding windows.                                                |
| **window_length**          | `int`           | 32               | All        | Length of each time-series window.                                                              |
| **stride**                 | `int`           | 16               | All        | Step between window start positions.                                                            |
| **noise**                  | `bool`          | False            | All        | Whether to add Gaussian noise.                                                                  |
| **noise_level**            | `float`         | 0.001            | All        | Standard deviation of added noise (relative to data scale).                                     |
| **standarize**             | `bool`          | False            | All        | If True, zero-mean/unit-variance scale each feature.                                             |
| **normalization**          | `bool`          | False            | All        | If True, Min-Max normalization.                                                            |
| **scale**                  | `bool`          | False            | All        | Whether to randomly scale specified channels.                                                   |
| **scale_range**            | `tuple(float)`  | (0.5, 2)       | All        | Uniform range from which to draw scale factors.                                                |
| **accelerations_to_scale** | `list[int]`     | [0,1,2,3,4,5]            | All        | Indices of channels to apply scaling.                                                           |
| **warp**                   | `bool`          | False            | All        | Whether to apply time-warp augmentation.                                                        |
| **warp_n_speed_change**    | `int`           | 2                | All        | Number of speed change segments in warp.                                                        |
| **warp_max_speed_ratio**   | `float`         | 2                | All        | Maximum relative speed multiplier in warp.                                                      |
| **shift**                  | `bool`          | False            | All        | Whether to randomly shift signals in time.                                                      |
| **shift_max**              | `int`           | 5                | All        | Maximum time-step shift magnitude (both directions).                                            |
| **feature_ablation_flag**  | `bool`          | False            | All        | If True, drops specified sensors/features.                                                      |
| **sensor_idx**             | `list[int]`     | []          | All        | Indices of sensors/features to remove when ablating.                                            |

### Model-Specific Architecture
| Name            | Type    | Default | Applies to      | Description                                  |
|-----------------|---------|---------|-----------------|----------------------------------------------|
| **hidden_size** | `int`   | 64      | LSTM            | Dimensionality of LSTM hidden state.         |
| **num_layers**  | `int`   | 1       | LSTM            | Number of stacked LSTM layers.               |
| **dropout**     | `float` | 0.3     | CNN, Heads      | Dropout probability for fully-connected layers. |
| **conv1_out**   | `int`   | —       | CNN/FCN/Heads   | Number of filters in first convolutional block. |
| **conv2_out**   | `int`   | —       | CNN/FCN/Heads   | Number of filters in second convolutional block. |
| **conv3_out**   | `int`   | —       | FCN, Heads      | Number of filters in third convolutional block. |
| **fc_hidden**   | `int`   | —       | CNN, Heads      | Size of the penultimate fully-connected layer. |

### Multi-Head Loss Balancing
| Name                      | Type       | Default | Applies to      | Description                                                  |
|---------------------------|------------|---------|-----------------|--------------------------------------------------------------|
| **lambda_phys**           | `int/float`| 1       | Physical Head   | Weight for the physical-feature head loss.                   |
| **phys_scale_factors**    | `list`/None| None    | Physical/Binary | [mass_factor, velocity_factor, decel_factor]. If set and **scale** is also True, it scales the scalars of each row by a determined factor, e.g if the signals are going to be amplified by 1.8 and phys_scale_factors = [0.5, 0.5, 0,5], the mass would be amplified by 1.8*0.5=0.9.  |
| **lambda_multi_head_loss**| `int/float`| 1       | Binary Head     | Weight for the binary-classification head loss.              |
