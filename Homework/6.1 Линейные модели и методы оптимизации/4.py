class MyElasticLogisticRegression(MyLogisticRegression):
    def __init__(self, l1_coef, l2_coef):
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.w = None

    def get_grad(self, X_batch, y_batch, predictions):
 

        grad_basic = X_batch.transpose() @ (predictions - y_batch)
        grad_l1 = self.l1_coef * np.sign(self.w)
        grad_l2 = 2 * self.l2_coef * self.w
        grad_l1[0] = 0
        grad_l2[0] = 0
        #Обнулять bias-компоненту вектора весов не нужно!

        assert grad_l1[0] == grad_l2[0] == 0, "Bias в регуляризационные слагаемые не входит!"
        assert grad_basic.shape == grad_l1.shape == grad_l2.shape == (X_batch.shape[1],)

        return grad_basic + grad_l1 + grad_l2
