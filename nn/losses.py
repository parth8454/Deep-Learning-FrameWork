class MSELoss:
    def __init__(self):
        pass

    def __call__(self,y_true,y_pred):
        losses = [(yout - ygt)**2 for ygt,yout in zip(y_true,y_pred)]

        total_loss = losses[0]
        for l in losses[1:]:
            total_loss += l

        return total_loss * (1.0 / len(y_true))