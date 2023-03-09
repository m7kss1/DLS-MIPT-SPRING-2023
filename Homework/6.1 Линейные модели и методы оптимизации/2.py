def generate_batches(X, y, batch_size):
    assert len(X) == len(y)
    np.random.seed(42)
    X = np.array(X)
    y = np.array(y)
    perm = np.random.permutation(len(X))
    num_batches = len(X) // batch_size 
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(len(X), (batch_idx + 1) * batch_size)
        yield X[perm[batch_start:batch_end]], y[perm[batch_start:batch_end]]
