import argparse


def parse_arg():
  argParser = argparse.ArgumentParser()
  argParser.add_argument("-t", "--train-report-interval", default=1, type=int, help="interval of logging epoch result during training")
  argParser.add_argument("-b", "--batch-size", default=64, type=int, help="batch size")
  argParser.add_argument("-d", "--data-path", default='./data/lg_600_data.npy', help="data path")
  argParser.add_argument("-p", "--show-progress-bar", action='store_true', help="show progress bar")
  argParser.set_defaults(show_progress_bar=False)
  argParser.add_argument("-r", "--resume-model-name", default=None, help="the result name under the results directory to resume training")
  argParser.add_argument("-s", "--save-epoch-interval", default=1000, type=int, help="epoch interval of saving model")
  argParser.add_argument("-n", "--result-notebook", default='train_lstm_gan.ipynb', help="result notebook name that will copy to result directory")

  argParser.add_argument("-e", "--gan-epochs", default=20000, type=int, help="number of epochs to train")
  argParser.add_argument("-gid", "--generator-input-dim", default=128, type=int, help="generator input dimension")
  argParser.add_argument("-dts", "--discriminator-train-steps", default=5, type=int, help="discriminator train steps")
  argParser.add_argument("-gp", "--gradient-penalty-lambda-term", default=10, type=float, help="gradient penalty lambda term")
  argParser.add_argument("-gdr", "--generator-dropout", default=0, type=float, help="generator dropout rate")
  argParser.add_argument("-glr", "--generator-learning-rate", default=0.0001, type=float, help="generator learning rate")
  argParser.add_argument("-gwd", "--generator-weight-decay", default=0, type=float, help="generator weight decay")
  argParser.add_argument("-gab1", "--generator-adam-beta1", default=0.5, type=float, help="generator adam beta1")
  argParser.add_argument("-gab2", "--generator-adam-beta2", default=0.999, type=float, help="generator adam beta2")
  argParser.add_argument("-ddr", "--discriminator-dropout", default=0, type=float, help="discriminator dropout rate")
  argParser.add_argument("-dlr", "--discriminator-learning-rate", default=0.0001, type=float, help="discriminator learning rate")
  argParser.add_argument("-dwd", "--discriminator-weight-decay", default=0, type=float, help="discriminator weight decay")
  argParser.add_argument("-dab1", "--discriminator-adam-beta1", default=0.5, type=float, help="discriminator adam beta1")
  argParser.add_argument("-dab2", "--discriminator-adam-beta2", default=0.999, type=float, help="discriminator adam beta2")
  argParser.add_argument("-ess", "--evaluate-sample-size", default=4000, type=int, help="the generate data sample size on evaluation")
  argParser.add_argument("-socs", "--soc-estimator-step", default=300, type=int, help="the soc estimator step size")
  argParser.add_argument("-socmp", "--soc-estimator-model-path", default='./soc_models/', help="the pre-trained soc estimator model path")
  argParser.add_argument("-socm", "--soc-estimator-model", default='2021-01-12-23-17-13_lstm_soc_percentage_lg_positive_temp_300_steps_mixed_cycle_test.h5', help="the pre-trained soc estimator model name")
  args = argParser.parse_args()
  
  return args