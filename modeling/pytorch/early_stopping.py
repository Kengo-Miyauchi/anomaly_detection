class EarlyStopping:
    def __init__(self, min_delta=1e-2, patience=10, verbose=0):
        self.epoch = 0
        self.pre_loss = float("inf")
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose

    def __call__(self, current_loss):
        if (self.pre_loss - current_loss) < self.min_delta:
            self.epoch += 1

            if self.epoch > self.patience:
                if self.verbose:
                    print("early stopping")
                return True

        else:
            self.epoch = 0
            self.pre_loss = current_loss

        return False
