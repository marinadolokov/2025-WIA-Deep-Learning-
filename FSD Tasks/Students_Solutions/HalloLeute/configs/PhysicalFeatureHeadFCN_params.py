# Model parameters:
params = dict(
  num_training_iterations = 3000,  # Increase this to train longer
  batch_size = 32,  # Experiment between 1 and 64
  learning_rate = 1e-4,  # Experiment between 1e-5 and 1e-1
  weight_decay = 0,  # Experiment between 1e-5 and 1e-1
  dropout=0.3,
  conv1_out=64,
  conv2_out=128,
  conv3_out=128,
  fc_hidden=64,
  NumberOfEpochs = 5, #Number of epochs

  #params for early stoppage
  use_early_stopping = True,
  patience = 10,
  min_delta = 1e-4,

  #Average experiments
  iterations = 2,

  #Physical features importance
  lambda_phys = 1,

  #Augmentation params
  augment_filter_fn = lambda row: row.label == 0,
  augment_filter_descrp = "lambda row: row.label == 0",

  #Examples:
  #lambda row: row.label > 0 just not working brakes
  #lambda row: random.random() < 0.3 just 30% of the data will be augmented randomly
  #lambda row: row.model == 'Variant Comfortline'

  truncation = True, truncation_portion = 0.3,
  oversample = True, oversample_fn = lambda row: row.label > 0, oversample_descrp = "lambda row: row.label > 0", oversample_times = 1,
  sliding = True, window_length = 32, stride = 16,
  noise = True, noise_level = 0.001,
  normalization = False,
  standarize = True,
  scale = True, scale_range = (0.8, 2.5), accelerations_to_scale = [3,4], phys_scale_factors = [1, 0.4, 0.2],
  warp = True, warp_n_speed_change = 2, warp_max_speed_ratio = 2,
  shift = True, shift_max = 5,

  # Feature ablation (Index of column or sensor type to drop)
  feature_ablation_flag = True,
  sensor_idx = [2,4,5],

  #Printing each epoch step
  print_epoch_stats = True

)