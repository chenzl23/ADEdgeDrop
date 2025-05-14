import argparse

def parameter_parser():
    parser = argparse.ArgumentParser(description = "Run .")

    parser.add_argument("--dataset-name", nargs = "?", default = "citeseer")
    parser.add_argument("--epoch-num", type = int, default = 200, help = "Number of training epochs. Default is 200.")
    parser.add_argument("--seed", type = int, default = 42, help = "Random seed for train-test split. Default is 42.")
    parser.add_argument("--dropout", type = float, default = 0.5, help = "Dropout parameter. Default is 0.5.")
    parser.add_argument("--learning-rate", type = float, default = 0.01, help = "Learning rate. Default is 0.01.")


    parser.add_argument("--verbose", type = int, default = 1, help = "Show training details.")

    parser.add_argument("--num-train-per-class", type = int, default = 20, help = "Train data num. Default is 20.")
    parser.add_argument("--num-val", type = int, default = 500, help = "Valid data num. Default is 500.")
    parser.add_argument("--num-test", type = int, default = 1000, help = "Test data num. Default is 1000.")
    
    parser.add_argument("--feature-normalize", type = int, default = 1, help = "If feature normalization")

    parser.add_argument("--drop-line-graph", type = float, default = 0.0, help = "If Drop line graph")
    parser.add_argument("--layer-num", type = int, default = 2, help = "Layer number.")
    parser.add_argument("--hidden-dim", type = int, default = 16, help = "Layer number.")

    parser.add_argument("--early-stop", type = bool, default = False, help = "If early stop")
    parser.add_argument("--patience", type = int, default = 20, help = "Patience for early stop")

    parser.add_argument('--adjmode', type=str, default='DAD', help='{DA, DAD}')


    parser.add_argument("--thred", type = float, default = 0.7, help = "Thred for edge mask")
    parser.add_argument("--alpha", type = float, default = 0.7, help = "Weights for updating edge attr")
    parser.add_argument("--m", type = int, default = 5, help = "m")
    parser.add_argument("--gamma", type = float, default = 0.7, help = "Step size for perturb")

    parser.add_argument("--backbone", type = str, default = "GCN", help = "Backbone network selection")

    parser.add_argument("--del-edge", type = float, default = 0, help = "Edge deletion probability")
    parser.add_argument("--add-edge", type = float, default = 0, help = "Edge addition probability")

    
    return parser.parse_args()
